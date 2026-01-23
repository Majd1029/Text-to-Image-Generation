import streamlit as st
from inference.pipeline import load_pipeline, generate_image
from utils.image_utils import save_image

st.set_page_config(page_title="Text-to-Image Generator", layout="centered")

st.title("🎨 Text-to-Image Generator")
st.write("Generate images using a fine-tuned Stable Diffusion model")

# Load model once
@st.cache_resource
def load_model():
    return load_pipeline()

pipe = load_model()

# --- User Input ---
prompt = st.text_input(
    "Enter your prompt",
    placeholder="e.g., A futuristic city at sunset, ultra-detailed"
)

steps = st.slider("Inference Steps", 10, 50, 30)
guidance = st.slider("Guidance Scale (CFG)", 1.0, 15.0, 7.5)

# --- Generate Button ---
if st.button("Generate Image"):
    if not prompt.strip():
        st.warning("Please enter a prompt.")
    else:
        with st.spinner("Generating image..."):
            image = generate_image(
                pipe,
                prompt=prompt,
                steps=steps,
                guidance=guidance
            )

            st.image(image, caption=prompt, use_container_width=True)
            path = save_image(image)
            st.success(f"Image saved to {path}")