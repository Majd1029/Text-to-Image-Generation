"""Streamlit interface for the Stable Diffusion pipeline.

The UI states which model is actually loaded. The LoRA has not been trained yet
(the first run failed, see README), so on a clean checkout this is base
Stable Diffusion v1.5 and the page says so rather than claiming a fine-tune.
"""
import streamlit as st

from inference.pipeline import (
    LORA_PATH,
    MODEL_ID,
    SETTINGS,
    generate_image,
    lcm_enabled,
    load_pipeline,
)
from utils.image_utils import save_image
from utils.prompt_utils import clean_prompt

st.set_page_config(page_title="Text-to-Image Generator", page_icon="🎨", layout="centered")

st.title("🎨 Text-to-Image Generator")


@st.cache_resource(show_spinner=False)
def load_model():
    return load_pipeline()


with st.spinner("Loading pipeline (first run downloads several GB)..."):
    pipe, using_lora = load_model()

if using_lora:
    st.caption(f"Stable Diffusion v1.5 with LoRA weights from `{LORA_PATH}`.")
else:
    st.warning(
        f"Running **base {MODEL_ID}** — no LoRA weights found at `{LORA_PATH}`. "
        "Images below are base Stable Diffusion output, not a fine-tune. "
        "See the README for training status."
    )

prompt = st.text_input(
    "Prompt", placeholder="e.g. a lighthouse in a storm, dramatic sky"
)

use_lcm = lcm_enabled()
limits = SETTINGS[use_lcm]
col1, col2, col3 = st.columns(3)
steps = col1.slider("Steps", *limits["steps"])
guidance = col2.slider("Guidance (CFG)", *limits["guidance"])
seed = col3.number_input("Seed (-1 = random)", value=-1, step=1)

if use_lcm:
    st.caption("LCM-LoRA is on: 4-8 steps is enough. Set USE_LCM=0 to turn it off.")
else:
    st.caption(
        "On CPU a 512px image takes several minutes. A CUDA GPU takes seconds."
    )

if st.button("Generate", type="primary"):
    cleaned = clean_prompt(prompt)
    if not cleaned:
        st.warning("Enter a prompt first.")
    else:
        with st.spinner("Generating..."):
            image = generate_image(
                pipe, prompt=cleaned, steps=steps, guidance=guidance, seed=seed
            )
        st.image(image, caption=cleaned, use_container_width=True)
        st.caption(
            f"Generated with {'LoRA fine-tune' if using_lora else 'base ' + MODEL_ID}"
        )
        st.success(f"Saved to {save_image(image)}")

st.divider()
st.caption(
    "[Source](https://github.com/Majd1029/Text-to-Image-Generation) · "
    "Dataset: 2,163 validated web-harvested image-caption pairs"
)
