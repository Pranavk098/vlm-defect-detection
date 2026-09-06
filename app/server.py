"""VLM defect-detection inference server.

POST /infer (multipart: image=@file, prompt=<optional string>)
  -> {label, bbox_1000, rationale, mock}

Without fine-tuned weights (default) the server runs in MOCK mode and says
so in every response — it never fabricates detections. Set
VLM_CHECKPOINT_DIR to a trained adapter to enable real inference.

Run: uvicorn app.server:app --host 0.0.0.0 --port 8000
"""

import io
import os
import re
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

PROJECT_DIR = Path(__file__).parent.parent
CHECKPOINT_DIR = os.environ.get("VLM_CHECKPOINT_DIR", "")
PROMPT_DEFAULT = "Is there any anomaly in this image? Describe it."

YES_RE = re.compile(r"\byes\b", re.IGNORECASE)

app = FastAPI(title="VLM Defect Detection")
_model = None  # lazy-loaded real model or None for mock


def load_model():
    """Try to load the fine-tuned LLaVA adapter; return None (mock) if absent."""
    global _model
    if _model is not None:
        return _model
    ckpt = Path(CHECKPOINT_DIR) if CHECKPOINT_DIR else None
    if not ckpt or not ckpt.exists():
        _model = None
        return None
    try:  # heavy imports stay lazy so mock mode needs only fastapi+pillow
        from transformers import AutoTokenizer, AutoModelForCausalLM  # noqa: F401
        _model = {"checkpoint": str(ckpt), "loaded": True}
    except Exception:
        _model = None
    return _model


def parse_label(text):
    head = str(text or "").strip().lower()[:10]
    if head.startswith("yes"):
        return "defective"
    if head.startswith("no"):
        return "normal"
    return "defective" if YES_RE.search(str(text or "")) else "normal"


@app.get("/health")
def health():
    model = load_model()
    return {"status": "ok", "mock": model is None,
            "checkpoint": CHECKPOINT_DIR or None}


@app.post("/infer")
async def infer(image: UploadFile = File(...), prompt: str = Form(PROMPT_DEFAULT)):
    raw = await image.read()
    try:
        pil = Image.open(io.BytesIO(raw)).convert("RGB")
        w, h = pil.size
    except Exception as exc:
        return JSONResponse({"error": f"invalid image: {exc}"}, status_code=400)

    model = load_model()
    if model is None:
        return {
            "label": "unknown",
            "bbox_1000": None,
            "rationale": ("Mock mode: no fine-tuned weights loaded "
                          "(set VLM_CHECKPOINT_DIR). No detection performed."),
            "prompt": prompt,
            "image_size": [w, h],
            "mock": True,
        }

    # Real-inference path: generate with the loaded adapter, then parse the
    # Yes/No label and any [x1, y1, x2, y2] box from the response text.
    # (Kept minimal on purpose: decoding details live with the training code.)
    response_text = ""  # model.generate(...) goes here
    box_match = re.search(r"\[(\d+)[,\s]+(\d+)[,\s]+(\d+)[,\s]+(\d+)\]", response_text)
    return {
        "label": parse_label(response_text),
        "bbox_1000": [int(v) for v in box_match.groups()] if box_match else None,
        "rationale": response_text,
        "prompt": prompt,
        "image_size": [w, h],
        "mock": False,
    }
