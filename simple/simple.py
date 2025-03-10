from diffusers import DiffusionPipeline
import torch

# Load the Stable Diffusion pipeline
pipeline = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", torch_dtype=torch.float32)
pipeline = pipeline.to("mps")

# Generate an image
prompt = "A majestic castle in the clouds"
image = pipeline(prompt).images[0]

# Save or display the image
image.save("majestic_castle.png")
