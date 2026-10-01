"""Download the model weights into the Hugging Face cache.

Run at Docker build time so the Space starts from a warm cache instead of
re-downloading ~4 GB every time it wakes up from sleep.
"""
from diffusers import StableDiffusionPipeline
from huggingface_hub import snapshot_download

from inference.pipeline import LCM_LORA_ID, MODEL_ID

if __name__ == "__main__":
    # Fetches only the files the pipeline loads, not the multi-GB .ckpt files
    # that also live in the model repo.
    StableDiffusionPipeline.download(MODEL_ID)
    snapshot_download(LCM_LORA_ID)
    print(f"Downloaded {MODEL_ID} and {LCM_LORA_ID}")
