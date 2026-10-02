"""Stable Diffusion pipeline, with LoRA applied only if weights actually exist.

Two corrections to the original:

* `runwayml/stable-diffusion-v1-5` no longer resolves — that org removed its
  model repos. The community mirror is `stable-diffusion-v1-5/stable-diffusion-v1-5`.
* `pipe.load_lora_weights(LORA_PATH)` was called unconditionally against a path
  under the gitignored `models/`, so it raised on any clean checkout. Worse, the
  training run that was meant to fill that directory failed, so the weights have
  never existed. Loading is now conditional and the caller can tell which model
  it actually got.

Optional LCM-LoRA speed-up: `latent-consistency/lcm-lora-sdv1-5` lets SD1.5
produce a usable image in 4-8 steps instead of 25-50, which is what makes a
CPU-only deployment (e.g. a free Hugging Face Space) practical. It is on by
default when no GPU is present; USE_LCM=1 / USE_LCM=0 forces it either way.
"""
import os
from pathlib import Path

import torch
from diffusers import LCMScheduler, StableDiffusionPipeline

MODEL_ID = os.getenv("MODEL_ID", "stable-diffusion-v1-5/stable-diffusion-v1-5")

# Local directory or a Hub repo id. Defaults to the repo the training notebook
# publishes to; until it holds weights, the base model is used. Set LORA_PATH=""
# to force the base model.
LORA_PATH = os.getenv("LORA_PATH", "MA29/t2i-lora")
LORA_WEIGHTS = "pytorch_lora_weights.safetensors"

LCM_LORA_ID = os.getenv("LCM_LORA_ID", "latent-consistency/lcm-lora-sdv1-5")

# (min, max, default) for each mode. LCM needs few steps and low guidance;
# higher guidance burns the image out.
SETTINGS = {
    True: {"steps": (2, 8, 4), "guidance": (1.0, 2.0, 1.0)},
    False: {"steps": (10, 50, 30), "guidance": (1.0, 15.0, 7.5)},
}


def gpu_available() -> bool:
    """True with a CUDA device, and on Hugging Face ZeroGPU Spaces, where the
    GPU is only attached while a @spaces.GPU function runs (so CUDA can look
    unavailable at load time) and SPACES_ZERO_GPU=true is set instead."""
    zero_gpu = os.getenv("SPACES_ZERO_GPU", "").strip().lower() in {"1", "true"}
    return zero_gpu or torch.cuda.is_available()


def lcm_enabled() -> bool:
    """USE_LCM forces it on or off; unset means on for CPU, off for GPU."""
    value = os.getenv("USE_LCM", "").strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return not gpu_available()


def lora_available(path: str = LORA_PATH) -> bool:
    """True only if real adapter weights exist, in a local directory or a Hub
    repo. An empty directory or repo does not count — that is exactly the state
    the failed training run left behind."""
    if not path:
        return False
    p = Path(path)
    if p.is_dir():
        return any(p.glob("*.safetensors")) or any(p.glob("*.bin"))
    if p.exists() or path.count("/") != 1:
        return False
    # Looks like a Hub repo id ("owner/name").
    try:
        from huggingface_hub import file_exists

        return file_exists(path, LORA_WEIGHTS)
    except Exception:  # offline, repo missing or private: fall back to base model
        return False


def load_pipeline(lora_path: str = LORA_PATH, use_lcm: bool | None = None):
    """Returns (pipe, using_lora). Callers should surface `using_lora` rather
    than claiming a fine-tune they may not have."""
    if use_lcm is None:
        use_lcm = lcm_enabled()
    device = "cuda" if gpu_available() else "cpu"
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)

    if device == "cuda":
        pipe.enable_attention_slicing()

    adapters = []
    if use_lcm:
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        pipe.load_lora_weights(LCM_LORA_ID, adapter_name="lcm")
        adapters.append("lcm")

    using_lora = False
    if lora_available(lora_path):
        pipe.load_lora_weights(lora_path, adapter_name="style")
        adapters.append("style")
        using_lora = True

    if len(adapters) > 1:
        # Both adapters stay active; loading a second one does not do this alone.
        pipe.set_adapters(adapters, adapter_weights=[1.0] * len(adapters))

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
