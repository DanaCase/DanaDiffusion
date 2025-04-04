import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import json
import os

class StableDiffusionDataset(Dataset):
    def __init__(
            self,
            image_dir,
            caption_file,
            instance_prompt,
            token_abstraction_dict,
            train_text_encoder_ti=True,
            size=512,
    ):
        self.image_dir = image_dir
        self.captions = json.load(open(caption_file, "r"))
        self.instance_prompt = instance_prompt
        self.token_abstraction_dict = token_abstraction_dict
        self.train_text_encoder_ti = train_text_encoder_ti

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
        image = self.transform(image)
        image.to("mps", dtype=torch.float32)

        caption = self.captions[image_name]
        for token_abs, token_replacement in self.token_abstraction_dict.items():
            caption = caption.replace(token_abs, "".join(token_replacement))
            

        return image, caption 

