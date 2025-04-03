import torch
import torch.nn.functional as F
from torch import nn, optim
from diffusers import (
        AutoencoderKL,
        StableDiffusionPipeline,
        DiffusionPipeline,
        UNet2DConditionModel,
        DDPMScheduler,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import free_memory
from transformers import AutoTokenizer, CLIPTextModel
from peft import LoraConfig, get_peft_model
import random
from PIL import Image
import os
from dotenv import load_dotenv

from simple.dataset import StableDiffusionDataset

load_dotenv()
DATA_DIR = os.getenv("DATA_DIR")
CAPTIONS_FILE = os.getenv("CAPTIONS_FILE")

MODEL_NAME = "runwayml/stable-diffusion-v1-5"
DTYPE = torch.float32

LR = 5e-5
BATCH_SIZE = 1
EPOCHS = 400

TEST_IMAGE_EVERY = 10
TEST_PROMPT = "A cute sks dog sitting on a bed"
TEST_IMAGE_DIR = "test_images"
TEST_SEED = 42



def encode_prompt(text_encocer, input_ids):
    text_input_ids = input_ids.to(text_encoder.device)

    prompt_embeds = text_encoder(
        text_input_ids,
        return_dict=False,
    )

    prompt_embeds = prompt_embeds[0]
    return prompt_embeds


device = "mps" if torch.backends.mps.is_available() else "cpu"

# load model components individually
scheduler = DDPMScheduler.from_pretrained(
        MODEL_NAME,
        subfolder="scheduler",
)
tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        subfolder="tokenizer",
        use_fast=False,
)
text_encoder = CLIPTextModel.from_pretrained(
        MODEL_NAME,
        subfolder="text_encoder"
)
vae = AutoencoderKL.from_pretrained(
        MODEL_NAME,
        subfolder="vae",
)
unet = UNet2DConditionModel.from_pretrained(
        MODEL_NAME,
        subfolder="unet",
)

# push to gpu
unet.to(device, dtype=DTYPE)
vae.to(device, dtype=DTYPE)
text_encoder.to(device, dtype=DTYPE)

# setup lora layers to target attention
unet_lora_config = LoraConfig(
    r=8,
    lora_alpha=8,
    init_lora_weights="gaussian",
    target_modules=["to_k", "to_q", "to_v"]
)
unet.add_adapter(unet_lora_config)

params_to_optimize = list(filter(lambda p: p.requires_grad, unet.parameters()))

# uncomment to see trained layers
# for name, param in unet.named_parameters():
    # if param.requires_grad:
        # print(name)

optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=LR,
)

# for some reason all my test images were flagged nsfw
# pipe.safety_checker = lambda images, clip_input: (images, [False])

train_dataset = StableDiffusionDataset(
    tokenizer=tokenizer,
    image_dir=DATA_DIR,
    caption_file=CAPTIONS_FILE
)
train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
)

# Define how often to generate test images (e.g., every 10 epochs)
os.makedirs(TEST_IMAGE_DIR, exist_ok=True)


# 📌 Step 5: Training Loop
steps_per_epoch = len(train_dataset)  # I think this only works for batch size 1
steps_for_scheduler = EPOCHS * steps_per_epoch

lr_scheduler = get_scheduler(
    "constant",
    optimizer=optimizer,
    num_training_steps=steps_for_scheduler,
)

for epoch in range(EPOCHS):
    unet.train()
    
    for step, batch in enumerate(train_dataloader):
        # load pixel data
        img = batch[0].to(device, DTYPE)
        input = vae.encode(img).latent_dist.sample()
        input = input * vae.config.scaling_factor
        
        # Sample noise
        noise = torch.randn_like(input)
        bsz, channels, height, width = input.shape

        # Sample a random timestep from each image
        timesteps = torch.randint(
            0, scheduler.config.num_train_timesteps, (bsz,), device=device)
        timesteps = timesteps.long()

        noisy_input = scheduler.add_noise(input, noise, timesteps)

        encoder_hidden_states = encode_prompt(
            text_encoder,
            batch[1]
        )
  
        model_pred = unet(
            noisy_input,
            timesteps,
            encoder_hidden_states,
            return_dict=False
        )[0]

        target = noise

        loss = F.mse_loss(model_pred, target, reduction="mean")
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        # print(f"loss: {loss.detach().item()} lr: {lr_scheduler.get_last_lr()[0]}")

    if (epoch) % TEST_IMAGE_EVERY == 0:
        pipeline = DiffusionPipeline.from_pretrained(
            MODEL_NAME,
            unet=unet,
            text_encoder=text_encoder,
            torch_dtype=DTYPE
        ).to(device, dtype=DTYPE)
        generator = torch.Generator(device=device).manual_seed(TEST_SEED)
        # with torch.amp.autocast(device):
        image = pipeline(
            prompt=TEST_PROMPT,
            num_inference_steps=25,
            generator=generator,
        ).images[0]
        image_path = os.path.join(TEST_IMAGE_DIR, f"epoch_{epoch+1}.png")
        image.save(image_path)
        del pipeline
        free_memory()





# 📌 Save LoRA Weights
# torch.save(pipe.unet.state_dict(), "lora_unet.pth")

# # 📌 Test Inference
# pipe.unet.load_state_dict(torch.load("lora_unet.pth"))
# pipe.unet.eval()
# test_prompt = "A cute sks dog sitting in a park"
# image = pipe(test_prompt).images[0]
# image.save("lora_output.png")

