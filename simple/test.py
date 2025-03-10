from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to("mps")
image = pipe("A cute dog sitting on a bed").images[0]
image.show()

