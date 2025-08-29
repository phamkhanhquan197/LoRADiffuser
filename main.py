# main.py
import torch
from diffusers import StableDiffusionPipeline
from lora_diffusion import load_safeloras, inject_trainable_lora

# -------------------------------
# 1. Setup
# -------------------------------
model_name = "runwayml/stable-diffusion-v1-5"
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# -------------------------------
# 2. Load Stable Diffusion pipeline
# -------------------------------
pipe = StableDiffusionPipeline.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
).to(device)

# -------------------------------
# 3. Load your trained LoRA(s)
# -------------------------------
lora_paths = "./exps/output_dsn/final_lora.safetensors"
loras = load_safeloras(lora_paths)

# Inject LoRA into UNet
inject_trainable_lora(pipe.unet, loras)

# -------------------------------
# 4. Generate an image
# -------------------------------
prompt = "A beautiful painting of a sunflower in a vase, highly detailed"
with torch.autocast(device):
    image = pipe(prompt, height=512, width=512, num_inference_steps=50).images[0]

# -------------------------------
# 5. Save the generated image
# -------------------------------
image.save("generated_image.png")
print("Image saved as generated_image.png")
