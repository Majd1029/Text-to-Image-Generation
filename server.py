"""HTTP API for the Hugging Face Space, used by the web page in `web/`.

    GET  /health    model status: "loading", "ready" or "error"
    POST /generate  {"prompt", "steps"?, "guidance"?, "seed"?} -> image/png

The model loads in a background thread so the server answers /health straight
away (the Space would otherwise look dead for the minute or two that loading
takes). A CPU Space can only run one generation at a time, so requests queue
behind a lock and anything beyond MAX_QUEUE gets a 429 instead of waiting
indefinitely.

Run locally:  uvicorn server:app --port 7860
"""
import io
import os
import random
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel, Field

from utils.prompt_utils import clean_prompt

MAX_PROMPT_CHARS = 300
MAX_QUEUE = int(os.getenv("MAX_QUEUE", "3"))

# Comma-separated list of sites allowed to call the API from a browser,
# e.g. "https://text-to-image.vercel.app". "*" allows any site.
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()]

state = {"status": "loading", "error": None, "pipe": None, "using_lora": False, "use_lcm": False}
generate_lock = threading.Lock()
queue_lock = threading.Lock()
waiting = 0


def _load_model():
    # Imported here so the server starts (and /health answers) before torch loads.
    from inference import pipeline

    try:
        use_lcm = pipeline.lcm_enabled()
        pipe, using_lora = pipeline.load_pipeline(use_lcm=use_lcm)
        state.update(pipe=pipe, using_lora=using_lora, use_lcm=use_lcm, status="ready")
    except Exception as exc:  # surfaced through /health
        state.update(status="error", error=f"{type(exc).__name__}: {exc}")
        raise


@asynccontextmanager
async def lifespan(_app):
    threading.Thread(target=_load_model, daemon=True).start()
    yield


app = FastAPI(title="Text-to-Image API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
    expose_headers=["X-Seed", "X-Steps", "X-Guidance", "X-Model"],
)

def _model_label() -> str:
    from inference.pipeline import MODEL_ID

    label = MODEL_ID
    if state["use_lcm"]:
        label += " + LCM-LoRA"
    if state["using_lora"]:
        label += " + fine-tuned LoRA"
    return label


@app.get("/", response_class=HTMLResponse)
def index():
    return (
        "<!doctype html><meta charset=utf-8><title>Text-to-Image API</title>"
        "<body style='font-family:system-ui;max-width:640px;margin:48px auto;padding:0 16px'>"
        "<h1>Text-to-Image API</h1>"
        "<p>This Space serves the API behind the Text-to-Image demo. "
        "<code>GET /health</code> reports model status; <code>POST /generate</code> returns a PNG.</p>"
        "<p><a href='https://github.com/Majd1029/Text-to-Image-Generation'>Source on GitHub</a></p>"
    )


@app.get("/health")
def health():
    body = {"status": state["status"], "queue": waiting}
    if state["status"] == "ready":
        from inference.pipeline import SETTINGS

        body.update(
            model=_model_label(),
            lcm=state["use_lcm"],
            fine_tuned=state["using_lora"],
            settings=SETTINGS[state["use_lcm"]],
        )
    elif state["status"] == "error":
        body["error"] = state["error"]
    return body


class GenerateRequest(BaseModel):
    prompt: str = Field(..., max_length=MAX_PROMPT_CHARS)
    steps: int | None = None
    guidance: float | None = None
    seed: int | None = Field(default=None, ge=-1, le=2**32 - 1)


def _clamp(value, limits):
    low, high, default = limits
    if value is None:
        return default
    return max(low, min(high, value))


@app.post("/generate")
def generate(req: GenerateRequest):
    global waiting
    if state["status"] == "loading":
        raise HTTPException(503, "The model is still loading. Try again in a minute.")
    if state["status"] == "error":
        raise HTTPException(500, "The model failed to load.")

    prompt = clean_prompt(req.prompt)
    if not prompt:
        raise HTTPException(422, "Enter a prompt.")

    from inference.pipeline import SETTINGS, generate_image

    limits = SETTINGS[state["use_lcm"]]
    steps = int(_clamp(req.steps, limits["steps"]))
    guidance = float(_clamp(req.guidance, limits["guidance"]))
    seed = req.seed if req.seed is not None and req.seed >= 0 else random.randint(0, 2**32 - 1)

    with queue_lock:
        if waiting >= MAX_QUEUE:
            raise HTTPException(429, "The demo is busy. Try again in a minute.")
        waiting += 1
    try:
        with generate_lock:
            image = generate_image(state["pipe"], prompt=prompt, steps=steps, guidance=guidance, seed=seed)
    finally:
        with queue_lock:
            waiting -= 1

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return Response(
        content=buf.getvalue(),
        media_type="image/png",
        headers={
            "X-Seed": str(seed),
            "X-Steps": str(steps),
            "X-Guidance": f"{guidance:g}",
            "X-Model": _model_label(),
            "Cache-Control": "no-store",
        },
    )
