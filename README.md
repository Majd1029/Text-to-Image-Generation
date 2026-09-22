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
| Inference with LoRA weights | Blocked on training |
| Streamlit app | Runs against base SD1.5 |

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
├── app.py                               # Streamlit interface
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

---

## Inference

```python
from inference.pipeline import load_pipeline, generate_image

pipe = load_pipeline()            # LoRA is optional; skipped if absent
image = generate_image(pipe, "a lighthouse in a storm", steps=30)
```

```bash
streamlit run app.py
```

Set the `LORA_PATH` environment variable to a local directory or a Hub repo id
once weights exist.

---

## Deployment note

A live generation demo does **not** fit a free hosting tier. SD1.5's UNet alone
is 3.44 GB; Streamlit Community Cloud allows ~2.7 GB for the whole app. The
intended demo is a gallery of images rendered on GPU and served as static files,
with generation done offline.

---

## Requirements

`diffusers`, `transformers`, `torch`, `accelerate`, `peft`, `datasets`,
`streamlit`, `pillow`.

Training needs a CUDA GPU. Inference runs on CPU, slowly — minutes per image at
512px.

---

## License

Educational and research use. The dataset is web-harvested: check image
copyright and licensing before redistributing anything derived from it.
