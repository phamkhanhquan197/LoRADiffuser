# evaluate_with_clip.py
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline
from transformers import CLIPProcessor, CLIPModel
import os

def evaluate_lora_with_clip(pipe, prompts, height=512, width=512, num_inference_steps=50, save_dir="./eval_results"):
    """
    Generates images using a LoRA-injected pipeline and computes CLIP similarity scores.

    Args:
        pipe: StableDiffusionPipeline with LoRA injected.
        prompts: list of strings, prompts for image generation.
        height: int, image height.
        width: int, image width.
        num_inference_steps: int, diffusion steps.
        save_dir: directory to save generated images.

    Returns:
        List of (image, clip_score) tuples.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Load CLIP model and processor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    generated_results = []
    pipe = pipe.to(device)

    for i, prompt in enumerate(prompts):
        with torch.autocast(device_type=device):
            image = pipe(prompt, height=height, width=width, num_inference_steps=num_inference_steps).images[0]
            image.save(f"{save_dir}/image_{i}.png")
            print(f"Saved image {i} for prompt: '{prompt}'")

        # Compute CLIP similarity
        inputs = clip_processor(text=[prompt], images=image, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            outputs = clip_model(**inputs)
            # Use cosine similarity between text & image embeddings
            image_emb = outputs.image_embeds
            text_emb = outputs.text_embeds
            clip_score = torch.cosine_similarity(image_emb, text_emb)
        
        generated_results.append((image, clip_score.item()))
        print(f"CLIP similarity for prompt '{prompt}': {clip_score.item():.4f}")

    return generated_results


# ------------------------
# Example usage
# ------------------------
if __name__ == "__main__":
    from lora_diffusion import load_safeloras, inject_trainable_lora

    # Load base pipeline
    model_name = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float16).to("cuda")

    # Inject LoRA
    lora_path = "./exps/output_dsn/final_lora.safetensors"
    lora = load_safeloras(lora_path)
    inject_trainable_lora(pipe.unet, lora)

    # Prompts to evaluate
    test_prompts = [
        "A fantasy castle in the mountains, cinematic lighting",
        "A cute cartoon dog riding a skateboard",
        "A futuristic city skyline at sunset, cyberpunk style, neon lights reflecting on wet streets",
        "A serene forest with a river, morning fog, ultra-realistic, high detail",
        "A whimsical illustration of a cat wizard casting a spell, colorful, cartoon style",
        "A majestic dragon flying over snowy mountains, cinematic lighting, epic fantasy",
        "A steampunk airship sailing through the clouds, intricate mechanical details",
        "A photorealistic portrait of an elderly woman smiling, soft lighting, 8k resolution",
        "A surreal painting of floating islands with waterfalls, dreamy and magical atmosphere",
        "A robot chef cooking in a modern kitchen, realistic lighting, cinematic angle",
        "A magical library with floating books and glowing runes, fantasy concept art",
        "A vibrant street market in Tokyo at night, bustling crowd, neon signs, rainy weather"
    ]


    results = evaluate_lora_with_clip(pipe, test_prompts)
    for i, (img, score) in enumerate(results):
        print(f"Image {i} CLIP score: {score:.4f}")
