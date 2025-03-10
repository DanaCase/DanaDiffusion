import torch
import torch.nn.functional as F
from torch import nn, optim
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import random
from PIL import Image
import os
from dotenv import load_dotenv

from diffusers import DDPMScheduler

from simple.dataset import StableDiffusionDataset

load_dotenv()
DATA_DIR = os.getenv("DATA_DIR")
CAPTIONS_FILE = os.getenv("CAPTIONS_FILE")

scheduler = DDPMScheduler(
    num_train_timesteps=1000,  # Standard for Stable Diffusion
    beta_start=0.00085, 
    beta_end=0.012, 
    beta_schedule="scaled_linear",
    clip_sample=True,
)


# 📌 Step 1: Load Stable Diffusion
device = "mps" if torch.backends.mps.is_available() else "cpu"
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to(device)

# for some reason all my test images were flagged nsfw
pipe.safety_checker = lambda images, clip_input: (images, [False])

# 📌 Step 2: Apply LoRA to U-Net Cross-Attention
lora_config = LoraConfig(
    r=4,
    lora_alpha=8,
    lora_dropout=0.1,
    target_modules=r"mid_block.attentions.*.transformer_blocks.*attn.*to_(q|k|v)" 
   )
pipe.unet = get_peft_model(pipe.unet, lora_config)

pipe.unet.print_trainable_parameters()

## Explicitly freeze non-lora layers
for param in pipe.unet.parameters():
    param.requires_grad = False  # Freeze everything

for name, param in pipe.unet.named_parameters():
    if "lora" in name:
        param.requires_grad = True  # Only train LoRA


# for name, param in pipe.unet.named_parameters():
    # if "lora" in name:
        # print(name, param.requires_grad)
# exit()

# 📌 Step 3: Load Dataset (Replace with Your Own)
dataset = dataset = StableDiffusionDataset(image_dir=DATA_DIR, caption_file=CAPTIONS_FILE)

train_data = [dataset.__getitem__(x) for x in range(9)]
train_images, train_prompts = zip(*train_data)

# 📌 Step 4: Define Optimizer
optimizer = optim.Adam(pipe.unet.parameters(), lr=5e-5)

# Define how often to generate test images (e.g., every 10 epochs)
TEST_IMAGE_EVERY = 10
TEST_PROMPT = "A cute sks dog sitting on a bed"

# Directory to save test images
TEST_IMAGE_DIR = "test_images"
os.makedirs(TEST_IMAGE_DIR, exist_ok=True)

# 📌 Step 5: Training Loop
epochs = 400
batch_size = 1

for epoch in range(epochs):
    epoch_loss = 0

    for i in range(len(train_images) // batch_size):
        optimizer.zero_grad()

        # 📌 Select random image-prompt pair
        idx = random.randint(0, len(train_images) - 1)
        img = train_images[idx]
        prompt = train_prompts[idx]

        # 📌 Encode Prompt
        # 📌 Step 1: Tokenize Prompt
        text_inputs = pipe.tokenizer(prompt, return_tensors="pt", padding="max_length", max_length=77).to(device)

        # 📌 Step 2: Get CLIP Text Embeddings
        text_embeddings = pipe.text_encoder(text_inputs.input_ids)[0]  # Extract embeddings
        
        # 📌 Generate Noisy Latents
        image_tensor = pipe.feature_extractor(img, return_tensors="pt").pixel_values.to(device)
        latents = pipe.vae.encode(image_tensor).latent_dist.sample()
        latents = latents * pipe.scheduler.init_noise_sigma

        # 📌 Add Noise
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, pipe.scheduler.config.num_train_timesteps, (1,), dtype=torch.long, device=device)

        
        alpha_prod_t = scheduler.alphas_cumprod.to(device)[timesteps].view(-1, 1, 1, 1)
        noisy_latents = alpha_prod_t.sqrt() * latents + (1 - alpha_prod_t).sqrt() * noise
        # noisy_latents = latents + noise
        # noisy_latents = noisy_latents.to(torch.float32)
    
        # 📌 Predict Noise using U-Net
        noise_pred = pipe.unet(noisy_latents, timesteps, text_embeddings).sample

        # 📌 Compute Loss
        loss = F.mse_loss(noise_pred, noise)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    # ✅ Generate a test image every TEST_IMAGE_EVERY epochs
    if (epoch) % TEST_IMAGE_EVERY == 0:
        print(f"Generating test image at epoch {epoch+1}...")
        pipe.unet.eval()
        # Generate image from the test prompt
        with torch.no_grad():
            image = pipe(TEST_PROMPT).images[0]  # Generate image
            test_pred = pipe.unet(noisy_latents, timesteps, text_embeddings).sample
            print("Noise Prediction Mean:", test_pred.mean().item())
            print("Noise Prediction Std:", test_pred.std().item())
       
       # Save image
        image_path = os.path.join(TEST_IMAGE_DIR, f"epoch_{epoch+1}.png")
        image.save(image_path)
        pipe.unet.train()

        print(f"✅ Test image saved to {image_path}")

    print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}")

# 📌 Save LoRA Weights
torch.save(pipe.unet.state_dict(), "lora_unet.pth")

# 📌 Test Inference
pipe.unet.load_state_dict(torch.load("lora_unet.pth"))
pipe.unet.eval()
test_prompt = "A cute sks dog sitting in a park"
image = pipe(test_prompt).images[0]
image.save("lora_output.png")

