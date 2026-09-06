"""Gradio demo: 15-class MVTec gallery with label + bbox + rationale.

Shows one sample image per MVTec category and calls the inference server
(app/server.py POST /infer); falls back to displaying the reference answer
and ground-truth box when the server is in mock mode.

Run: python app/demo.py  (demo on :7860, expects API on :8000 or set VLM_API_URL)
"""

import io
import json
import os
from pathlib import Path

import gradio as gr
from PIL import Image, ImageDraw

PROJECT_DIR = Path(__file__).parent.parent
API_URL = os.environ.get("VLM_API_URL", "http://localhost:8000")

CATEGORIES_15 = [
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
    "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor",
    "wood", "zipper",
]


def sample_for(category):
    """First defective test image for a category, else first good one."""
    test_dir = PROJECT_DIR / "mvtec_anomaly_detection" / category / "test"
    if not test_dir.exists():
        return None, None
    for defect_dir in sorted(test_dir.iterdir()):
        if defect_dir.is_dir() and defect_dir.name != "good":
            imgs = sorted(defect_dir.glob("*.png"))
            if imgs:
                return imgs[0], defect_dir.name
    good = test_dir / "good"
    imgs = sorted(good.glob("*.png")) if good.exists() else []
    return (imgs[0], "good") if imgs else (None, None)


def ground_truth_box(image_path):
    """Look up the grounding-subset box for an image path, if present."""
    gf = PROJECT_DIR / "mvtec_grounding_200.json"
    if not gf.exists():
        return None
    try:
        items = json.loads(gf.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    rel = "/".join(Path(str(image_path)).parts[-3:])
    for item in items:
        if item.get("image") == rel:
            return item.get("bbox_1000")
    return None


def draw_box(pil_img, box):
    if not box:
        return pil_img
    img = pil_img.copy()
    w, h = img.size
    x1, y1, x2, y2 = [box[0] / 1000 * w, box[1] / 1000 * h,
                      box[2] / 1000 * w, box[3] / 1000 * h]
    ImageDraw.Draw(img).rectangle([x1, y1, x2, y2], outline="red", width=3)
    return img


def query_api(pil_img, prompt):
    try:
        import requests
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        resp = requests.post(f"{API_URL}/infer",
                             files={"image": ("input.png", buf.getvalue(), "image/png")},
                             data={"prompt": prompt}, timeout=60)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


def predict(category):
    img_path, defect = sample_for(category)
    if img_path is None:
        return None, "Dataset not found — mount mvtec_anomaly_detection/ first."
    pil = Image.open(img_path).convert("RGB")
    prompt = ("Locate the defect as [x1, y1, x2, y2] (0-1000 scale) "
              "and explain in one sentence.")
    result = query_api(pil, prompt)
    if result and not result.get("mock"):
        label, box = result.get("label", "?"), result.get("bbox_1000")
        rationale = result.get("rationale", "")
    else:  # mock / offline fallback: reference answer, honestly labeled
        label = "defective" if defect != "good" else "normal"
        box = ground_truth_box(img_path)
        rationale = (f"Reference (no model): {defect} "
                     f"{'— ground-truth box shown' if box else '(no mask available)'}.")
        label += " [reference]"
    return draw_box(pil, box), f"**{label}** — {rationale}"


with gr.Blocks(title="VLM Defect Detection — MVTec x15") as demo:
    gr.Markdown("# VLM Defect Detection — all 15 MVTec categories")
    category = gr.Dropdown(CATEGORIES_15, value="bottle", label="Category")
    run = gr.Button("Inspect sample")
    out_img = gr.Image(label="Sample + defect box")
    out_text = gr.Markdown()
    run.click(predict, inputs=category, outputs=[out_img, out_text])
    gr.Examples([[c] for c in CATEGORIES_15], inputs=category)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
