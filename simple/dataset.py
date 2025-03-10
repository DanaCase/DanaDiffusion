import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import json
import os

class StableDiffusionDataset(Dataset):
    def __init__(self, image_dir, caption_file, size=512):
        self.image_dir = image_dir
        self.captions = json.load(open(caption_file, "r"))

        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # Normalize for Stable Diffusion VAE
        ])

        self.image_filenames = list(self.captions.keys())

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_name = self.image_filenames[idx]
        image_path = os.path.join(self.image_dir, image_name)
        
        image = Image.open(image_path).convert("RGB")
        # image = self.transform(image)

        caption = self.captions[image_name]

        return image, caption

