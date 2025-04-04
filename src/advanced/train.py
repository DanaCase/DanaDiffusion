from typing import Any, Dict, Optional, Union, List
import copy
import re
import math
import itertools
import transformers
import torch
import torch.nn.functional as F
from torch import nn, optim

from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from PIL import Image
from PIL.ImageOps import exif_transpose
from safetensors.torch import save_file
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import crop
from tqdm.auto import tqdm
from transformers import (  
    CLIPTokenizer,
    PretrainedConfig,
    T5TokenizerFast,
    AutoTokenizer,
    CLIPTextModel,
    T5EncoderModel,
)
import diffusers
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    FluxPipeline,
    FluxTransformer2DModel,
    FlowMatchEulerDiscreteScheduler,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    _set_state_dict_into_text_encoder,
    cast_training_params,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
    free_memory,
)
from diffusers.utils import (
    check_min_version,
    convert_unet_state_dict_to_peft,
    is_wandb_available,
)
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.torch_utils import is_compiled_module

from dotenv import load_dotenv
import os

from advanced.dataset import StableDiffusionDataset

load_dotenv()
DATA_DIR = os.getenv("DATA_DIR")
CAPTIONS_FILE = os.getenv("CAPTIONS_FILE")

MODEL_NAME = "black-forest-labs/FLUX.1-dev"
DTYPE = torch.float32

LR = 5e-5
LRSCHEDULER = "constant"
LRWARMUP_STEPS = 500
LR_CYCLES = 1
LR_POWER = 1.0
BATCH_SIZE = 1
EPOCHS = 400
ADAM_WEIGHT_DECAY = 1e-04
EPS = 1e-08

WEIGHTING_SCHEME = None
GUIDANCE_SCALE = 3.5
MAX_GRAD_NORM = 1

ADAM_WDECAY_TEXT_ENCODER = 1e-03
TEXT_ENCODER_LR = 5e-6
INITALIZER_CONCEPT = 'a dog'
TRAIN_TEXT_ENCODER_TI_FRAC = 0.5
TOKEN_ABSTRACTION = "sks"
INSTANCE_PROMPT = "photo of an sks dog"
MAX_PROMPT_LENGTH = 512 

B1 = 0.9
B2 = 0.999

TEST_IMAGE_EVERY = 10
TEST_PROMPT = "A cute sks dog sitting on a bed"
TEST_IMAGE_DIR = "test_images"
TEST_SEED = 42


device = "mps" if torch.backends.mps.is_available() else "cpu"

class CustomFlowMatchEulerDiscreteScheduler(FlowMatchEulerDiscreteScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        with torch.no_grad():
            # create weights for timesteps
            num_timesteps = 1000

            # generate the multiplier based on cosmap loss weighing
            # this is only used on linear timesteps for now

            # cosine map weighing is higher in the middle and lower at the ends
            # bot = 1 - 2 * self.sigmas + 2 * self.sigmas ** 2
            # cosmap_weighing = 2 / (math.pi * bot)

            # sigma sqrt weighing is significantly higher at the end and lower at the beginning
            sigma_sqrt_weighing = (self.sigmas**-2.0).float()
            # clip at 1e4 (1e6 is too high)
            sigma_sqrt_weighing = torch.clamp(sigma_sqrt_weighing, max=1e4)
            # bring to a mean of 1
            sigma_sqrt_weighing = sigma_sqrt_weighing / sigma_sqrt_weighing.mean()

            # Create linear timesteps from 1000 to 0
            timesteps = torch.linspace(1000, 0, num_timesteps, device="cpu")

            self.linear_timesteps = timesteps
            # self.linear_timesteps_weights = cosmap_weighing
            self.linear_timesteps_weights = sigma_sqrt_weighing

            # self.sigmas = self.get_sigmas(timesteps, n_dim=1, dtype=torch.float32, device='cpu')
            pass

    def get_weights_for_timesteps(self, timesteps: torch.Tensor) -> torch.Tensor:
        # Get the indices of the timesteps
        step_indices = [(self.timesteps == t).nonzero().item() for t in timesteps]

        # Get the weights for the timesteps
        weights = self.linear_timesteps_weights[step_indices].flatten()

        return weights

    def get_sigmas(self, timesteps: torch.Tensor, n_dim, dtype, device) -> torch.Tensor:
        sigmas = self.sigmas.to(device=device, dtype=dtype)
        schedule_timesteps = self.timesteps.to(device)
        timesteps = timesteps.to(device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)

        return sigma

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        ## ref https://github.com/huggingface/diffusers/blob/fbe29c62984c33c6cf9cf7ad120a992fe6d20854/examples/dreambooth/train_dreambooth_sd3.py#L1578
        ## Add noise according to flow matching.
        ## zt = (1 - texp) * x + texp * z1

        # sigmas = get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)
        # noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise

        # timestep needs to be in [0, 1], we store them in [0, 1000]
        # noisy_sample = (1 - timestep) * latent + timestep * noise
        t_01 = (timesteps / 1000).to(original_samples.device)
        noisy_model_input = (1 - t_01) * original_samples + t_01 * noise

        # n_dim = original_samples.ndim
        # sigmas = self.get_sigmas(timesteps, n_dim, original_samples.dtype, original_samples.device)
        # noisy_model_input = (1.0 - sigmas) * original_samples + sigmas * noise
        return noisy_model_input

    def scale_model_input(self, sample: torch.Tensor, timestep: Union[float, torch.Tensor]) -> torch.Tensor:
        return sample

    def set_train_timesteps(self, num_timesteps, device, linear=False):
        if linear:
            timesteps = torch.linspace(1000, 0, num_timesteps, device=device)
            self.timesteps = timesteps
            return timesteps
        else:
            # distribute them closer to center. Inference distributes them as a bias toward first
            # Generate values from 0 to 1
            t = torch.sigmoid(torch.randn((num_timesteps,), device=device))

            # Scale and reverse the values to go from 1000 to 0
            timesteps = (1 - t) * 1000

            # Sort the timesteps in descending order
            timesteps, _ = torch.sort(timesteps, descending=True)

            self.timesteps = timesteps.to(device=device)

            return timesteps


# load model components individually
scheduler = CustomFlowMatchEulerDiscreteScheduler.from_pretrained(
        MODEL_NAME,
        subfolder="scheduler",
)
noise_scheduler_copy = copy.deepcopy(scheduler)

tokenizer1 = CLIPTokenizer.from_pretrained(
        MODEL_NAME,
        subfolder="tokenizer",
        use_fast=False,
)
tokenizer2 = T5TokenizerFast.from_pretrained(
        MODEL_NAME,
        subfolder="tokenizer_2",
        use_fast=False,
)

text_encoder1 = CLIPTextModel.from_pretrained(
        MODEL_NAME,
        subfolder="text_encoder"
)
text_encoder2 = T5EncoderModel.from_pretrained(
        MODEL_NAME,
        subfolder="text_encoder_2"
)
vae = AutoencoderKL.from_pretrained(
        MODEL_NAME,
        subfolder="vae",
)
transformer = FluxTransformer2DModel.from_pretrained(
        MODEL_NAME,
        subfolder="transformer",
)


# dont trian by default
transformer.requires_grad_(False)
vae.requires_grad_(False)
text_encoder1.requires_grad_(False)
text_encoder2.requires_grad_(False)

# push to gpu
transformer.to(device, dtype=DTYPE)
vae.to(device, dtype=DTYPE)
text_encoder1.to(device, dtype=DTYPE)
text_encoder2.to(device, dtype=DTYPE)

# setup lora layers to target attention
transformer_lora_config = LoraConfig(
    r=8,
    lora_alpha=8,
    init_lora_weights="gaussian",
    target_modules=[
            "attn.to_k",
            "attn.to_q",
            "attn.to_v",
            "attn.to_out.0",
            "attn.add_k_proj",
            "attn.add_q_proj",
            "attn.add_v_proj",
            "attn.to_add_out",
            "ff.net.0.proj",
            "ff.net.2",
            "ff_context.net.0.proj",
            "ff_context.net.2",
        ]
)
transformer.add_adapter(transformer_lora_config)
transformer_lora_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))

# setup text encoder training
# pivotal tuning training only CLIP for now
token_abstraction_list = [placeholder.strip() for placeholder in re.split(r",\s*", TOKEN_ABSTRACTION)]
print(f"list of token identifiers: {token_abstraction_list}")
token_ids = tokenizer1.encode(INITALIZER_CONCEPT, add_special_tokens=False)
num_new_tokens_per_abstraction = len(token_ids)

token_abstraction_dict = {}
token_idx = 0
for i, token in enumerate(token_abstraction_list):
    token_abstraction_dict[token] = [f"<s{token_idx + i + j}>" for j in range(num_new_tokens_per_abstraction)]
    token_idx += num_new_tokens_per_abstraction - 1

for token_abs, token_replacement in token_abstraction_dict.items():
    new_instance_prompt = INSTANCE_PROMPT.replace(token_abs, "".join(token_replacement))

validation_prompt = TEST_PROMPT.replace(token_abs, "".join(token_replacement))

text_encoders = [text_encoder1, text_encoder2]
tokenizers = [tokenizer1, tokenizer2]



class TokenEmbeddingsHandler:
    def __init__(self, text_encoders, tokenizers):
        self.text_encoders = text_encoders
        self.tokenizers = tokenizers

        self.train_ids: Optional[torch.Tensor] = None
        self.train_ids_t5: Optional[torch.Tensor] = None
        self.inserting_toks: Optional[List[str]] = None
        self.embeddings_settings = {}
        self.initializer_concept = None

    def initialize_new_tokens(self, inserting_toks: List[str]):
        idx = 0
        for tokenizer, text_encoder in zip(self.tokenizers, self.text_encoders):
            assert isinstance(inserting_toks, list), "inserting_toks should be a list of strings."
            assert all(
                isinstance(tok, str) for tok in inserting_toks
            ), "All elements in inserting_toks should be strings."

            self.inserting_toks = inserting_toks
            special_tokens_dict = {"additional_special_tokens": self.inserting_toks}
            tokenizer.add_special_tokens(special_tokens_dict)
            # Resize the token embeddings as we are adding new special tokens to the tokenizer
            text_encoder.resize_token_embeddings(len(tokenizer))

            # Convert the token abstractions to ids
            if idx == 0:
                self.train_ids = tokenizer.convert_tokens_to_ids(self.inserting_toks)
            else:
                self.train_ids_t5 = tokenizer.convert_tokens_to_ids(self.inserting_toks)

            # random initialization of new tokens
            embeds = (
                text_encoder.text_model.embeddings.token_embedding if idx == 0 else text_encoder.encoder.embed_tokens
            )
            std_token_embedding = embeds.weight.data.std()

            print(f"{idx} text encoder's std_token_embedding: {std_token_embedding}")

            train_ids = self.train_ids if idx == 0 else self.train_ids_t5
            # if initializer_concept are not provided, token embeddings are initialized randomly
            if self.initializer_concept is None:
                hidden_size = (
                    text_encoder.text_model.config.hidden_size if idx == 0 else text_encoder.encoder.config.hidden_size
                )
                embeds.weight.data[train_ids] = (
                    torch.randn(len(train_ids), hidden_size).to(device=self.device).to(dtype=self.dtype)
                    * std_token_embedding
                )
            else:
                # Convert the initializer_token, placeholder_token to ids
                initializer_token_ids = tokenizer.encode(self.initializer_concept, add_special_tokens=False)
                for token_idx, token_id in enumerate(train_ids):
                    embeds.weight.data[token_id] = (embeds.weight.data)[
                        initializer_token_ids[token_idx % len(initializer_token_ids)]
                    ].clone()

            self.embeddings_settings[f"original_embeddings_{idx}"] = embeds.weight.data.clone()
            self.embeddings_settings[f"std_token_embedding_{idx}"] = std_token_embedding

            # makes sure we don't update any embedding weights besides the newly added token
            index_no_updates = torch.ones((len(tokenizer),), dtype=torch.bool)
            index_no_updates[train_ids] = False

            self.embeddings_settings[f"index_no_updates_{idx}"] = index_no_updates

            print(self.embeddings_settings[f"index_no_updates_{idx}"].shape)

            idx += 1

    def save_embeddings(self, file_path: str):
        assert self.train_ids is not None, "Initialize new tokens before saving embeddings."
        tensors = {}
        # text_encoder_one, idx==0 - CLIP ViT-L/14, text_encoder_two, idx==1 - T5 xxl
        idx_to_text_encoder_name = {0: "clip_l", 1: "t5"}
        for idx, text_encoder in enumerate(self.text_encoders):
            train_ids = self.train_ids if idx == 0 else self.train_ids_t5
            embeds = text_encoder.text_model.embeddings.token_embedding if idx == 0 else text_encoder.shared
            assert embeds.weight.data.shape[0] == len(self.tokenizers[idx]), "Tokenizers should be the same."
            new_token_embeddings = embeds.weight.data[train_ids]

            # New tokens for each text encoder are saved under "clip_l" (for text_encoder 0),
            # Note: When loading with diffusers, any name can work - simply specify in inference
            tensors[idx_to_text_encoder_name[idx]] = new_token_embeddings
            # tensors[f"text_encoders_{idx}"] = new_token_embeddings

        save_file(tensors, file_path)

    @property
    def dtype(self):
        return self.text_encoders[0].dtype

    @property
    def device(self):
        return self.text_encoders[0].device

    def retract_embeddings(self):
        print(self.text_encoders)
        for idx, text_encoder in enumerate(self.text_encoders):
            embeds = text_encoder.text_model.embeddings.token_embedding if idx == 0 else text_encoder.shared
            index_no_updates = self.embeddings_settings[f"index_no_updates_{idx}"]
            embeds.weight.data[index_no_updates] = (
                self.embeddings_settings[f"original_embeddings_{idx}"][index_no_updates]
                .to(device=text_encoder.device)
                .to(dtype=text_encoder.dtype)
            )

            # for the parts that were updated, we need to normalize them
            # to have the same std as before
            std_token_embedding = self.embeddings_settings[f"std_token_embedding_{idx}"]

            index_updates = ~index_no_updates
            new_embeddings = embeds.weight.data[index_updates]
            off_ratio = std_token_embedding / new_embeddings.std()

            new_embeddings = new_embeddings * (off_ratio**0.1)
            embeds.weight.data[index_updates] = new_embeddings


embedding_handler = TokenEmbeddingsHandler(text_encoders, tokenizers)
inserting_toks = []
for new_tok in token_abstraction_dict.values():
    inserting_toks.extend(new_tok)
embedding_handler.initialize_new_tokens(inserting_toks=inserting_toks)


text_lora_config = LoraConfig(
        r=8,
        lora_alpha=8,
        init_lora_weights="gaussian",
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        )
text_encoder1.add_adapter(text_lora_config)

text_lora_parameters_one = []
for name, param in text_encoder1.named_parameters():
    if "token_embedding" in name:
        param.data = param.to(dtype=torch.float32)
        param.requires_grad = True
        text_lora_parameters_one.append(param)
    else:
        param.requires_grad = False


# Set up the optimizer
transformer_parameters_with_lr = {"params": transformer_lora_parameters, "lr": LR}
text_parameters_one_with_lr = {
            "params": text_lora_parameters_one,
            "weight_decay": ADAM_WDECAY_TEXT_ENCODER,
            "lr": TEXT_ENCODER_LR,
        }
params_to_optimize = [transformer_parameters_with_lr, text_parameters_one_with_lr]
te_idx = 1

optimizer_class = torch.optim.AdamW
optimizer = optimizer_class(
    params_to_optimize,
    betas=(B1, B2),
    weight_decay=ADAM_WEIGHT_DECAY,
    eps=EPS,
)

add_special_tokens_clip = True
add_special_tokens_t5 = False

vae_config_shift_factor = vae.config.shift_factor
vae_config_scaling_factor = vae.config.scaling_factor
vae_config_block_out_channels = vae.config.block_out_channels

weight_dtype = torch.float32


def log_validation(
    pipeline,
    pipeline_args,
    epoch,
    is_final_validation=False
):
    pipeline = pipeline.to(device, dtype=DTYPE)
    generator = torch.Generator(device=device).manual_seed(TEST_SEED)
    image = pipeline(
        prompt=TEST_PROMPT,
        num_inference_steps=25,
        generator=generator,
    ).images[0]
    image_path = os.path.join(TEST_IMAGE_DIR, f"epoch_{epoch+1}.png")
    image.save(image_path)



def tokenize_prompt(tokenizer, prompt, max_sequence_length, add_special_tokens=False):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        return_length=False,
        return_overflowing_tokens=False,
        add_special_tokens=add_special_tokens,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    return text_input_ids


def _get_t5_prompt_embeds(
    text_encoder,
    tokenizer,
    max_sequence_length=512,
    prompt=None,
    num_images_per_prompt=1,
    device=None,
    text_input_ids=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("text_input_ids must be provided when the tokenizer is not specified")

    prompt_embeds = text_encoder(text_input_ids.to(device))[0]

    dtype = text_encoder.dtype
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape

    # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds


def _get_clip_prompt_embeds(
    text_encoder,
    tokenizer,
    prompt: str,
    device=None,
    text_input_ids=None,
    num_images_per_prompt: int = 1,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_overflowing_tokens=False,
            return_length=False,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("text_input_ids must be provided when the tokenizer is not specified")

    prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=False)

    # Use pooled output of CLIPTextModel
    prompt_embeds = prompt_embeds.pooler_output
    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)

    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, -1)

    return prompt_embeds

def encode_prompt(
    text_encoders,
    tokenizers,
    prompt: str,
    max_sequence_length,
    device=None,
    num_images_per_prompt: int = 1,
    text_input_ids_list=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)
    dtype = text_encoders[0].dtype

    pooled_prompt_embeds = _get_clip_prompt_embeds(
        text_encoder=text_encoders[0],
        tokenizer=tokenizers[0],
        prompt=prompt,
        device=device if device is not None else text_encoders[0].device,
        num_images_per_prompt=num_images_per_prompt,
        text_input_ids=text_input_ids_list[0] if text_input_ids_list is not None else None,
    )

    prompt_embeds = _get_t5_prompt_embeds(
        text_encoder=text_encoders[1],
        tokenizer=tokenizers[1],
        max_sequence_length=max_sequence_length,
        prompt=prompt,
        num_images_per_prompt=num_images_per_prompt,
        device=device if device is not None else text_encoders[1].device,
        text_input_ids=text_input_ids_list[1] if text_input_ids_list is not None else None,
    )

    text_ids = torch.zeros(batch_size, prompt_embeds.shape[1], 3).to(device=device, dtype=dtype)
    text_ids = text_ids.repeat(num_images_per_prompt, 1, 1)

    return prompt_embeds, pooled_prompt_embeds, text_ids


train_dataset = StableDiffusionDataset(
    image_dir=DATA_DIR,
    caption_file=CAPTIONS_FILE,
    train_text_encoder_ti=True,
    instance_prompt=new_instance_prompt,
    token_abstraction_dict=token_abstraction_dict,
)
train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
)

num_update_steps_per_epoch = math.ceil(len(train_dataloader) / 1)
max_training_steps = EPOCHS * num_update_steps_per_epoch

num_train_epochs_text_encoder = int(TRAIN_TEXT_ENCODER_TI_FRAC  * EPOCHS)
num_train_epochs_transformer = int(EPOCHS)


lr_scheduler = get_scheduler(
    LRSCHEDULER,
    optimizer=optimizer,
    num_warmup_steps=LRWARMUP_STEPS,
    num_training_steps=max_training_steps,
    num_cycles=LR_CYCLES,
    power=LR_POWER,
)

print("Training")
print(f"Num examples = {len(train_dataset)}")
print(f"Num batches = {len(train_dataloader)}")
print(f"Num epochs = {EPOCHS}")
print(f"Total optimization steps = {max_training_steps}")

global_step = 0
first_epoch = 0

progress_bar = tqdm(
    range(0, max_training_steps),
    initial=global_step,
    desc="Steps",
)
def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
    sigmas = noise_scheduler_copy.sigmas.to(device=device, dtype=DTYPE)
    schedule_timesteps = noise_scheduler_copy.timesteps.to(device)
    timesteps = timesteps.to(device)
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma


pivoted_te = False
pivoted_tr = False
for epoch in range(first_epoch, EPOCHS):
    transformer.train()
    
    if epoch == num_train_epochs_text_encoder:
       print(f"PIVOT TE {epoch}")
       pivoted_te = True
    else:
       text_encoder1.train()
    
    if epoch == num_train_epochs_transformer:
        print(f"PIVOT TRANSFORMER {epoch}")
        pivoted_tr = True

    for step, batch in enumerate(train_dataloader):
        if pivoted_te:
            optimizer.param_groups[te_idx]["lr"] = 0.0
            optimizer.param_groups[-1]["lr"] = 0.0
        elif pivoted_tr:
            optimizer.param_groups[0]["lr"] = 0.0

        prompts = batch[1]
        elems_to_repeat = 1

        # CLIP has max lenght of 77
        tokens_one = tokenize_prompt(
                tokenizer1,
                prompts,
                max_sequence_length=77,
                add_special_tokens=add_special_tokens_clip,
        )
        tokens_two = tokenize_prompt(
                tokenizer2,
                prompts,
                max_sequence_length=MAX_PROMPT_LENGTH,
                add_special_tokens=add_special_tokens_t5,
        )
        
        prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt(
            text_encoders=[text_encoder1, text_encoder2],
            tokenizers=[None, None],
            text_input_ids_list=[
                tokens_one.repeat(elems_to_repeat, 1),
                tokens_two.repeat(elems_to_repeat, 1),
            ],
            max_sequence_length=MAX_PROMPT_LENGTH,
            device=device,
            prompt=prompts,
        )
        
        pixel_values = batch[0].to(device, dtype=vae.dtype)
        model_input = vae.encode(pixel_values).latent_dist.sample()

        model_input = (model_input - vae_config_shift_factor) * vae_config_scaling_factor
        model_input = model_input.to(dtype=weight_dtype)

        vae_scale_factor = 2 ** (len(vae_config_block_out_channels))

        latent_image_ids = FluxPipeline._prepare_latent_image_ids(
            model_input.shape[0],
            model_input.shape[2] // 2,
            model_input.shape[3] // 2,
            device,
            weight_dtype
        )

        noise = torch.randn_like(model_input)
        bsz = model_input.shape[0]

        u = compute_density_for_timestep_sampling(
            weighting_scheme = WEIGHTING_SCHEME,
            batch_size=bsz
        )
        indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
        timesteps = noise_scheduler_copy.timesteps[indices].to(device=model_input.device)

        sigmas = get_sigmas(timesteps, n_dim=model_input.dim(), dtype=model_input.dtype)
        noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise

        packed_noisy_model_input = FluxPipeline._pack_latents(
            noisy_model_input,
            batch_size=model_input.shape[0],
            num_channels_latents=model_input.shape[1],
            height=model_input.shape[2],
            width=model_input.shape[3]
        )

        guidance = torch.tensor([GUIDANCE_SCALE], device=device)
        guidance = guidance.expand(model_input.shape[0])

        print(type(transformer.time_text_embed))
        print(transformer.time_text_embed.forward.__code__.co_varnames)

        model_pred = transformer(
            hidden_states=packed_noisy_model_input,
            timestep = timesteps / 1000,
            guidance=guidance,
            pooled_projections = pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            return_dict=False,
        )[0]

        model_pred = FluxPipeline._unpack_latents(
            model_pred,
            height=model_input.shape[2] * vae_scale_factor,
            width=model_input.shape[3] * vae_scale_factor,
            vae_scale_factor=vae_scale_factor,
        )

        weighting = compute_loss_weighting_for_sd3(weighting_scheme=WEIGHTING_SCHEME, sigmas=sigmas)

        # flow matching loss
        target = noise - model_input

        loss = torch.mean(
            (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
            1,
        )
        loss = loss.mean()
        loss.backward()

        params_to_clip = itertools.chain(transformer.parameters(), text_encoder1.parameters())
        torch.nn.utils.clip_grad_norm_(params_to_clip, MAX_GRAD_NORM)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        
        embedding_handler.retract_embeddings()

        progress_bar.update(1)
        global_step += 1
        
        print(f"loss {loss.detach().item()}")

        if global_step >= max_training_steps:
            break

        if epoch % TEST_IMAGE_EVERY == 0:
            pipeline = FluxPipeline.from_pretrained(
                MODEL_NAME,
                vae=vae,
                text_encoder=text_encoder1,
                text_endoder_2=text_encoder2,
                transformer=transformer,
                torch_dtype=weight_dtype,
            )

            pipeline_args = {"prompt": TEST_PROMPT}
            images = log_validation(
                pipeline=pipeline,
                pipeline_args=pipeline_args,
                epoch=epoch,
            )
            images = None
            del pipeline
            del text_encoder2
            free_memory()
