---
title: Text to Image
emoji: 🎨
colorFrom: purple
colorTo: pink
sdk: gradio
sdk_version: 6.29.0
app_file: space_app.py
pinned: false
short_description: Stable Diffusion v1.5 text-to-image on ZeroGPU
---

# 🎨 Stable Diffusion LoRA — dataset pipeline and training setup

Builds an image-caption dataset by harvesting and validating web images, then
fine-tunes Stable Diffusion v1.5 with LoRA and serves the result through a
Streamlit interface.

> **Status: the LoRA is not trained yet.** The first training run failed and
> produced no weights. The cause is identified and fixed — see
> [Training status](#training-status). Until that run completes, the inference
> code falls back to base Stable Diffusion v1.5 and says so. Nothing in this
> repository is currently a fine-tuned model.

---

## What actually works today

| Stage | State |
|---|---|
| Harvest image URLs + captions from a source CSV | Working |
| Validate URLs, drop dead links | Working — **2,163** valid pairs |
| Download images, write `valid_captions.csv` | Working |
| LoRA fine-tuning | **Fixed, not yet re-run** |
| Inference with base SD1.5 | Working |
| Inference with LoRA weights | Ready: loads `MA29/t2i-lora` automatically once training publishes it |
| Streamlit app | Runs against base SD1.5 |
| Hosted demo (Hugging Face Space + Vercel page) | Ready to deploy — see [Live demo and deployment](#live-demo-and-deployment) |

---

## Training status

The original training cell failed with:

```
subprocess.CalledProcessError: Command '[... train_text_to_image_lora.py ...]'
returned non-zero exit status 2
```

No weights were written, and the following cell continued with the base model
instead of stopping — which is why this went unnoticed. Three distinct causes:

1. **Wrong script path.** `train_text_to_image_lora.py` is in
   `diffusers/examples/text_to_image/`, not `examples/community/`.
2. **Invalid arguments.** `--train_annotation_file` and `--save_every_n_steps`
   do not exist in that script. Passing them is an argparse error, which would
   have failed even with the correct path. The script reads captions from a
   `metadata.jsonl` inside the image directory, not from a separate CSV.
3. **Dead base model.** `runwayml/stable-diffusion-v1-5` was removed from the
   Hub. The community mirror is `stable-diffusion-v1-5/stable-diffusion-v1-5`.

`train_lora_FIXED.ipynb` corrects all three, converts the captions CSV into the
expected `metadata.jsonl`, and asserts that weights exist before continuing —
so a silent failure cannot repeat.

---

## Repository layout

```
├── notebooks/
│   └── Text-to-Image-Generation.ipynb   # harvest, validate, download, train, infer
├── inference/
│   └── pipeline.py                      # base pipeline + optional LoRA
├── utils/
│   ├── image_utils.py
│   └── prompt_utils.py
├── web/
│   └── index.html                       # static demo page (deployed on Vercel)
├── app.py                               # Streamlit interface (local)
├── space_app.py                         # Gradio app for the Hugging Face Space
└── requirements.txt
```

`data/` and `models/` are gitignored. The dataset (565 MB `data.csv`,
`valid_images/`, `valid_captions.csv`) and any trained weights live outside the
repository.

---

## Dataset

A source CSV of image URLs and captions. The notebook samples from it, checks
each URL is still live, downloads what remains, and writes `valid_captions.csv`:

```
image_path, caption
```

**2,163** validated image-caption pairs. Captions are raw web alt-text — noisy,
and mostly stock-photo phrasing such as *"family of four hugging each other
stock photo"*. Worth knowing when judging what a fine-tune on this data can
reasonably be expected to learn.

---

## Training

```bash
accelerate launch --mixed_precision="fp16" \
  diffusers/examples/text_to_image/train_text_to_image_lora.py \
  --pretrained_model_name_or_path="stable-diffusion-v1-5/stable-diffusion-v1-5" \
  --train_data_dir="<dir containing images + metadata.jsonl>" \
  --caption_column="text" \
  --resolution=512 --random_flip \
  --train_batch_size=1 --gradient_accumulation_steps=4 \
  --max_train_steps=2000 --learning_rate=1e-04 \
  --lr_scheduler="cosine" --rank=4 \
  --checkpointing_steps=500 --seed=42 \
  --output_dir="<output>"
```

Roughly 1-2 hours on a free Colab T4. Output is
`pytorch_lora_weights.safetensors`, about 3 MB at rank 4.

The easy way is `train_lora_FIXED.ipynb` in Colab (*Runtime → Run all* on a T4).
It reads the dataset from Google Drive, resumes after a disconnect, and publishes
the weights to the Hub as `MA29/t2i-lora`. Restart the demo Space afterwards and
it loads the fine-tune on startup.

---

## Inference

```python
from inference.pipeline import load_pipeline, generate_image

pipe, using_lora = load_pipeline()   # LoRA is optional; skipped if absent
image = generate_image(pipe, "a lighthouse in a storm", steps=30)
```

```bash
streamlit run app.py
```

The LoRA is loaded from `LORA_PATH`: a local directory or a Hub repo id,
`MA29/t2i-lora` by default. If it holds no weights yet, the base model is used.
Set `LORA_PATH=""` to force the base model.

**LCM-LoRA speed-up.** On CPU the pipeline loads
[`latent-consistency/lcm-lora-sdv1-5`](https://huggingface.co/latent-consistency/lcm-lora-sdv1-5)
and switches to the LCM scheduler, so 4 steps at guidance 1.0 give a usable
image instead of 30 steps at 7.5. Set `USE_LCM=0` to turn it off, or
`USE_LCM=1` to use it on a GPU too.

---

## Live demo and deployment

The demo has two parts:

```
Vercel (web/index.html)  ──@gradio/client──▶  Hugging Face Space (space_app.py)
   static page, free                            SD1.5 on a free ZeroGPU
```

Streamlit Community Cloud can't host this: SD1.5's UNet alone is 3.44 GB and it
allows ~2.7 GB for the whole app. A ZeroGPU Space is free and attaches a GPU
only while an image is being generated. Visitors share a daily GPU quota; when
it runs out the page says so. Free Spaces sleep after a period without
visitors; the page shows "Waking up the server…" and waits while the Space
restarts.

### API (`space_app.py`)

| Endpoint | Inputs | Returns |
|---|---|---|
| `/generate` | `prompt`, `steps`, `guidance`, `seed` (-1 = random) | image, seed used |
| `/settings` | — | model label, slider ranges |

Call it with [`@gradio/client`](https://www.npmjs.com/package/@gradio/client)
(as `web/index.html` does) or the Python `gradio_client`. Run it locally with
`python space_app.py`.

### 1. Hugging Face Space

1. On huggingface.co, create a **new Space** → SDK **Gradio** → *Blank* →
   hardware **ZeroGPU**, visibility **Public**. Name it `text-to-image`.
2. Create an access token with **write** permission
   (Settings → Access Tokens).
3. In this GitHub repo, go to Settings → Secrets and variables → Actions and add:
   - secret `HF_TOKEN` = the token
   - variable `HF_SPACE` = `<hf-username>/text-to-image`, e.g. `MA29/text-to-image`
4. Run the **Sync to Hugging Face Space** workflow (Actions tab), or push to
   `main`. It mirrors this repo to the Space, which installs the requirements
   and starts `space_app.py`. The first start downloads the model (~4 GB).

### 2. Vercel page

1. On vercel.com, **Add New → Project** and import this repository.
2. Set **Root Directory** to `web`, Framework Preset **Other**, no build
   command. Deploy.
3. If the Space isn't `MA29/text-to-image`, change `window.SPACE_ID` at the
   top of `web/index.html`.

## Requirements

`diffusers`, `transformers`, `torch`, `accelerate`, `peft`, `datasets`,
`streamlit`, `pillow`.

Training needs a CUDA GPU. Inference runs on a GPU in seconds; on CPU it takes
minutes per image at 512px with the standard scheduler, much less with the
LCM-LoRA (on by default on CPU).

---

## License

Educational and research use. The dataset is web-harvested: check image
copyright and licensing before redistributing anything derived from it.
