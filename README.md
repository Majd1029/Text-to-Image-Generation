# 🎨 Stable Diffusion LoRA Fine-Tuning with Streamlit Interface

This project demonstrates how to **fine-tune Stable Diffusion using LoRA** on a custom web-harvested image-caption dataset and deploy an **interactive Streamlit app** for image generation.

The notebook (`genai.ipynb`) contains the full end-to-end pipeline, while the Streamlit app provides an easy-to-use interface for inference.

---

## 🚀 Project Overview

This project follows a complete Generative AI workflow:

1. **Dataset Loading** from a CSV file of image URLs and captions  
2. **URL Validation** to remove broken or invalid links  
3. **Image Downloading** and caption storage  
4. **LoRA Fine-Tuning** on Stable Diffusion v1.5  
5. **Inference Pipeline** combining base model + LoRA weights  
6. **Streamlit Web App** for prompt-based image generation  

---

## 📁 Project Structure

```bash
genai-stable-diffusion-lora/
│
├── genai.ipynb # Main notebook (full pipeline)
├── app.py # Streamlit interface
├── requirements.txt # Dependencies
├── README.md # Project documentation
│
├── data/
│ ├── raw/
│ │ └── data.csv
│ └── processed/
│ ├── valid_images/
│ └── valid_captions.csv
│
└── models/
└── lora_sd_v1_5/ # Trained LoRA weights
```


---

## 📊 Dataset Format

The dataset CSV should contain:

| image_url | caption |
|-----------|---------|
| URL to image | Text description of image |

---

## 🧠 Notebook Pipeline (`genai.ipynb`)

The notebook is the **core of the project** and includes:

### 1️⃣ Data Loading
Loads image URLs and captions from a CSV file.

### 2️⃣ URL Validation
Checks which image links are still active.

### 3️⃣ Image Downloading
Downloads valid images and saves them locally.

### 4️⃣ Dataset Preparation
Creates a `valid_captions.csv` file with:
```
image_path, caption
```

### 5️⃣ LoRA Training
Fine-tunes Stable Diffusion using Hugging Face Diffusers.

### 6️⃣ Inference
Loads:
- Base model: `runwayml/stable-diffusion-v1-5`
- Your trained LoRA weights  
Then generates images from prompts.

---

## 🏋️ Training LoRA

Training is launched from inside the notebook using:

```bash
accelerate launch diffusers/examples/community/train_text_to_image_lora.py ...
```
Training Output:
```bash
models/lora_sd_v1_5/
```

## 🎨 Streamlit Web App

The Streamlit app allows users to generate images from prompts using the fine-tuned model.

**Run the app**

First export the notebook as a Python module:
```bash
jupyter nbconvert --to python genai.ipynb
```

Then start Streamlit:
```bash
streamlit run app.py
```

## ⚙️ Installation
```bash
git clone https://github.com/yourusername/genai-stable-diffusion-lora.git
cd genai-stable-diffusion-lora

pip install -r requirements.txt
```

## 🖥 Requirements

Main libraries used:

- diffusers
- transformers
- torch
- accelerate
- datasets
- streamlit
- pillow


## 📌 Notes

- Training requires a GPU (recommended: CUDA-enabled)
- Inference can run on CPU but will be slower
- LoRA makes fine-tuning lightweight compared to full model training

## 📜 License

This project is for educational and research purposes.
Make sure your dataset usage follows image copyright and licensing rules.