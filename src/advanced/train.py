from typing import List, Optional, Union
import copy

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

from simple.dataset import StableDiffusionDataset

load_dotenv()
DATA_DIR = os.getenv("DATA_DIR")
CAPTIONS_FILE = os.getenv("CAPTIONS_FILE")

MODEL_NAME = "black-forest-labs/FLUX.1-dev"
DTYPE = torch.float32

LR = 5e-5
BATCH_SIZE = 1
EPOCHS = 400
ADAM_WEIGHT_DECAY = 1e-04
EPS = 1e-08

ADAM_WDECAY_TEXT_ENCODER = 1e-03
TEXT_ENCODER_LR = 5e-6 

B1 = 0.9
B2 = 0.999

TEST_IMAGE_EVERY = 10
TEST_PROMPT = "A cute sks dog sitting on a bed"
TEST_IMAGE_DIR = "test_images"
TEST_SEED = 42

TRAIN_TEXT_ENCODER_TI_FRAC=0.5

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
text_lora_config = LoraConfig(
        r=8,
        lora_alpha=8,
        init_lora_weights="gaussian",
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        )
text_encoder1.add_adapter(text_lora_config)
text_lora_parameters1 = list(filter(lambda p: p.requires_grad, text_encoder1.parameters()))

# Set up the optimizer
transformer_parameters_with_lr = {"params": transformer_lora_parameters, "lr": LR}
text_parameters_one_with_lr = {
            "params": text_lora_parameters1,
            "weight_decay": ADAM_WDECAY_TEXT_ENCODER,
            "lr": TEXT_ENCODER_LR,
        }
params_to_optimize = [transformer_parameters_with_lr, text_parameters_one_with_lr]

optimizer_class = torch.optim.AdamW
optimizer = optimizer_class(
    params_to_optimize,
    betas=(B1, B2),
    weight_decay=ADAM_WEIGHT_DECAY,
    eps=EPS,
)

def log_validation():
    # going to need to get both text encoders in there
        pipeline = FluxPipeline.from_pretrained(
            MODEL_NAME,
            torch_dtype=DTYPE
        ).to(device, dtype=DTYPE)
        generator = torch.Generator(device=device).manual_seed(TEST_SEED)
        image = pipeline(
            prompt=TEST_PROMPT,
            num_inference_steps=25,
            generator=generator,
        ).images[0]
        image_path = os.path.join(TEST_IMAGE_DIR, f"epoch_{epoch+1}.png")
        image.save(image_path)
        del pipeline
        free_memory()

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


