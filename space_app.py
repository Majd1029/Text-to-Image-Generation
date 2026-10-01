"""Gradio app for the Hugging Face Space (ZeroGPU), also used by web/index.html.

API (call with @gradio/client or gradio_client):
    /generate (prompt, steps, guidance, seed) -> (image, seed used)
    /settings ()                              -> model label and slider ranges

On ZeroGPU the GPU is attached only while a @spaces.GPU function runs, so the
pipeline is loaded at import time and only generation is decorated. Outside a
ZeroGPU Space the decorator does nothing and the app runs on whatever device
is available.
"""
import spaces  # must be imported before torch on ZeroGPU

import random

import gradio as gr

from inference.pipeline import MODEL_ID, SETTINGS, generate_image, lcm_enabled, load_pipeline
from utils.prompt_utils import clean_prompt

MAX_PROMPT_CHARS = 300
MAX_SEED = 2**32 - 1

USE_LCM = lcm_enabled()
pipe, USING_LORA = load_pipeline(use_lcm=USE_LCM)
LIMITS = SETTINGS[USE_LCM]
MODEL_LABEL = MODEL_ID + (" + LCM-LoRA" if USE_LCM else "") + (" + fine-tuned LoRA" if USING_LORA else "")


def _clamp(value, limits):
    low, high, default = limits
    if value is None:
        return default
    return max(low, min(high, value))


@spaces.GPU(duration=40)
def _run(prompt, steps, guidance, seed):
    return generate_image(pipe, prompt=prompt, steps=steps, guidance=guidance, seed=seed)


def generate(prompt: str, steps: float, guidance: float, seed: float):
    prompt = clean_prompt(prompt or "")
    if not prompt:
        raise gr.Error("Enter a prompt.")
    if len(prompt) > MAX_PROMPT_CHARS:
        raise gr.Error(f"Keep the prompt under {MAX_PROMPT_CHARS} characters.")
    steps = int(_clamp(steps, LIMITS["steps"]))
    guidance = float(_clamp(guidance, LIMITS["guidance"]))
    seed = int(seed) if seed is not None and 0 <= seed <= MAX_SEED else random.randint(0, MAX_SEED)
    return _run(prompt, steps, guidance, seed), seed


def settings() -> dict:
    return {
        "model": MODEL_LABEL,
        "lcm": USE_LCM,
        "fine_tuned": USING_LORA,
        "steps": list(LIMITS["steps"]),
        "guidance": list(LIMITS["guidance"]),
    }


with gr.Blocks(title="Text-to-Image Generator") as demo:
    gr.Markdown(
        "# Text-to-Image Generator\n"
        f"Model: **{MODEL_LABEL}**. "
        + ("" if USING_LORA else "The project's own fine-tuned LoRA hasn't been trained yet, so these are base-model images. ")
        + "[Source on GitHub](https://github.com/Majd1029/Text-to-Image-Generation)"
    )
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label="Prompt", lines=3, max_length=MAX_PROMPT_CHARS,
                                placeholder="e.g. a lighthouse on a cliff in a storm, dramatic sky, oil painting")
            with gr.Accordion("Advanced settings", open=False):
                steps = gr.Slider(LIMITS["steps"][0], LIMITS["steps"][1], value=LIMITS["steps"][2], step=1, label="Steps")
                guidance = gr.Slider(LIMITS["guidance"][0], LIMITS["guidance"][1], value=LIMITS["guidance"][2], step=0.1, label="Guidance")
                seed = gr.Number(value=-1, precision=0, label="Seed (-1 = random)")
            run = gr.Button("Generate", variant="primary")
            gr.Examples(
                examples=[
                    ["a cozy cabin in a snowy forest at dusk, warm light"],
                    ["an astronaut riding a horse, watercolor"],
                    ["a bowl of ramen, studio photo, shallow depth of field"],
                    ["a futuristic city skyline at night, neon, rain"],
                ],
                inputs=[prompt],
            )
        with gr.Column():
            image = gr.Image(label="Result", type="pil", format="png")
            seed_used = gr.Number(label="Seed used", precision=0, interactive=False)

    run.click(generate, inputs=[prompt, steps, guidance, seed], outputs=[image, seed_used], api_name="generate")
    gr.api(settings, api_name="settings")

demo.queue(max_size=20)

if __name__ == "__main__":
    demo.launch()
