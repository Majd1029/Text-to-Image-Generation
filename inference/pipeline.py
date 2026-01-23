import torch
from diffusers import StableDiffusionPipeline

MODEL_ID = "runwayml/stable-diffusion-v1-5"
LORA_PATH = "models/lora_weights"

def load_pipeline():
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )

    if torch.cuda.is_available():
        pipe = pipe.to("cuda")

    pipe.load_lora_weights(LORA_PATH)
    return pipe

def generate_image(pipe, prompt, steps=30, guidance=7.5):
    image = pipe(
        prompt,
        num_inference_steps=steps,
        guidance_scale=guidance
    ).images[0]

    return image