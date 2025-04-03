import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import json
import os

class StableDiffusionDataset(Dataset):
    def __init__(self, image_dir, caption_file, tokenizer, size=512,):
        self.image_dir = image_dir
        self.captions = json.load(open(caption_file, "r"))
        self.tokenizer = tokenizer

        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # Normalize for Stable Diffusion VAE
        ])

        self.image_filenames = list(self.captions.keys())

    def __len__(self):
        return len(self.image_filenames)

    def tokenize_prompt(self, prompt):
        max_length = self.tokenizer.model_max_length

        text_inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding="max_length",
                max_length=max_length
        )
        return text_inputs


    def __getitem__(self, idx):
        image_name = self.image_filenames[idx]
        image_path = os.path.join(self.image_dir, image_name)
        
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        image.to("mps", dtype=torch.float32)

        caption = self.captions[image_name]
        tokenized = self.tokenize_prompt(
                caption
        )

        return image, tokenized.input_ids

