"""Stable Diffusion pipeline, with LoRA applied only if weights actually exist.

Two corrections to the original:

* `runwayml/stable-diffusion-v1-5` no longer resolves — that org removed its
  model repos. The community mirror is `stable-diffusion-v1-5/stable-diffusion-v1-5`.
* `pipe.load_lora_weights(LORA_PATH)` was called unconditionally against a path
  under the gitignored `models/`, so it raised on any clean checkout. Worse, the
  training run that was meant to fill that directory failed, so the weights have
  never existed. Loading is now conditional and the caller can tell which model
  it actually got.
"""
import os
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline

MODEL_ID = os.getenv("MODEL_ID", "stable-diffusion-v1-5/stable-diffusion-v1-5")

# Local directory or a Hub repo id. Empty/missing means base model only.
LORA_PATH = os.getenv("LORA_PATH", "models/lora_sd_v1_5")


def lora_available(path: str = LORA_PATH) -> bool:
    """True only if the directory exists and holds real adapter weights.
    An empty directory does not count — that is exactly the state the failed
    training run left behind."""
    if not path:
        return False
    p = Path(path)
    if not p.is_dir():
        return False
    return any(p.glob("*.safetensors")) or any(p.glob("*.bin"))


def load_pipeline(lora_path: str = LORA_PATH):
    """Returns (pipe, using_lora). Callers should surface `using_lora` rather
    than claiming a fine-tune they may not have."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)

    if device == "cuda":
        pipe.enable_attention_slicing()

    using_lora = False
    if lora_available(lora_path):
        pipe.load_lora_weights(lora_path)
        using_lora = True

    pipe.set_progress_bar_config(disable=True)
    return pipe, using_lora


def generate_image(pipe, prompt, steps=30, guidance=7.5, seed=None):
    generator = None
    if seed is not None and seed >= 0:
        generator = torch.Generator(pipe.device.type).manual_seed(int(seed))
    return pipe(
        prompt,
        num_inference_steps=int(steps),
        guidance_scale=float(guidance),
        generator=generator,
    ).images[0]
