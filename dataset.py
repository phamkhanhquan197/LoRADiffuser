from datasets import load_dataset
import os
from PIL import Image

# Load a small sample dataset from HuggingFace
dataset = load_dataset("huggan/flowers-102-categories", split="train[:20]")

# Save directory for images
save_dir = "./data/flowers"
os.makedirs(save_dir, exist_ok=True)

# Save images to folder
for i, example in enumerate(dataset):
    img = example["image"].convert("RGB")
    img.save(f"{save_dir}/{i}.png")

print(f"Saved {len(dataset)} images to {save_dir}")
