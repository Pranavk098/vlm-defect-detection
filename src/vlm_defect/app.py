"""Gradio demo — LLaVA-1.5 fine-tuned for manufacturing defect detection.

Supports two loading modes:
  • HF Hub (default) — loads the merged model pushed by push_to_hub.py
  • Local LoRA checkpoint — loads base model + adapter from disk

Usage:
    # From HF Hub (after push_to_hub.py has been run):
    vlm-app --repo-id your-username/llava-mvtec-defect-detection

    # From a local LoRA checkpoint:
    vlm-app --checkpoint checkpoints/llava-mvtec-lora-v3/checkpoint-500 \\
            --config configs/local_8gb.yaml

    # Or run directly:
    python src/vlm_defect/app.py --repo-id your-username/llava-mvtec-defect-detection
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper",
]

# Per-category P(Yes) thresholds from the v3 checkpoint-500 sweep.
# These are the same values used in evaluate.py CATEGORY_THRESHOLDS.
CATEGORY_THRESHOLDS: dict[str, float] = {
    "bottle":     0.20,
    "cable":      0.25,
    "capsule":    0.30,
    "carpet":     0.15,
    "grid":       0.15,
    "hazelnut":   0.20,
    "leather":    0.40,
    "metal_nut":  0.50,
    "pill":       0.25,
    "screw":      0.40,
    "tile":       0.50,
    "toothbrush": 0.80,
    "transistor": 0.40,
    "wood":       0.30,
    "zipper":     0.40,
}

# Global fallback threshold (optimal from v3 sweep)
DEFAULT_THRESHOLD = 0.25


# ---------------------------------------------------------------------------
# Prompt builder — must match the training prompt format exactly
# ---------------------------------------------------------------------------

def _make_prompt(category: str) -> str:
    """Build the category-aware LLaVA prompt used during training."""
    return (
        f"USER: <image>\n"
        f"Is there any anomaly in this {category} image? "
        f"If yes, say 'Yes, there is a <defect_type> anomaly.' "
        f"If no, say 'No.' ASSISTANT:"
    )


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_from_hub(repo_id: str):
    """Load merged model + processor directly from HuggingFace Hub."""
    from transformers import AutoProcessor, LlavaForConditionalGeneration

    print(f"[INFO] Loading merged model from Hub: {repo_id}")
    processor = AutoProcessor.from_pretrained(repo_id)
    model = LlavaForConditionalGeneration.from_pretrained(
        repo_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model, processor


def _load_from_checkpoint(checkpoint: Path, config: Path):
    """Load base model + LoRA adapter from a local checkpoint."""
    import yaml
    from transformers import AutoProcessor, BitsAndBytesConfig, LlavaForConditionalGeneration
    from peft import PeftModel

    with open(config) as f:
        cfg = yaml.safe_load(f)
    base_model_id = cfg["model"]["name_or_path"]

    print(f"[INFO] Loading processor from {base_model_id}")
    processor = AutoProcessor.from_pretrained(base_model_id)

    print(f"[INFO] Loading base model {base_model_id} in 4-bit")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = LlavaForConditionalGeneration.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    print(f"[INFO] Loading LoRA adapter from {checkpoint}")
    model = PeftModel.from_pretrained(model, str(checkpoint.resolve()))
    model.eval()
    return model, processor


# ---------------------------------------------------------------------------
# Attention map extraction (CLIP vision encoder)
# ---------------------------------------------------------------------------

def _get_vision_tower(model):
    """Navigate PEFT/LLaVA wrappers to reach the CLIP vision tower."""
    m = model
    # Unwrap PeftModel → LlavaForConditionalGeneration
    if hasattr(m, "base_model"):
        m = m.base_model
    if hasattr(m, "model"):
        m = m.model
    # m is now LlavaForConditionalGeneration
    if hasattr(m, "model") and hasattr(m.model, "vision_tower"):
        return m.model.vision_tower
    return None


def compute_attention_overlay(
    image: Image.Image,
    model,
    processor,
    category: str,
) -> Image.Image | None:
    """Compute a saliency overlay using CLIP vision encoder attention maps.

    Extracts the last-layer multi-head attention from the CLIP ViT, averages
    across attention heads, takes the CLS-to-patch attention vector, reshapes
    it to a 2-D spatial map, and overlays it on the original image as a heat map.

    Returns None when the vision tower is inaccessible (e.g. some Hub models).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
    except ImportError:
        return None

    vision_tower = _get_vision_tower(model)
    if vision_tower is None:
        return None

    prompt = _make_prompt(category)
    try:
        inputs = processor(
            text=prompt,
            images=image,
            return_tensors="pt",
        ).to(model.device)
    except Exception:
        return None

    try:
        with torch.inference_mode():
            vision_out = vision_tower.vision_model(
                pixel_values=inputs["pixel_values"],
                output_attentions=True,
            )
    except Exception:
        return None

    if not vision_out.attentions:
        return None

    # Last layer attention: (1, n_heads, n_patches+1, n_patches+1)
    attn = vision_out.attentions[-1]           # last transformer block
    attn = attn.mean(dim=1)                    # avg over heads → (1, n+1, n+1)
    attn = attn[0, 0, 1:]                      # CLS→patch slice → (n_patches,)
    attn = attn.float().cpu()

    # For CLIP ViT-L/14@336: 336/14 = 24 patches per side → 576 total
    n = attn.shape[0]
    grid = int(n ** 0.5)
    if grid * grid != n:
        return None

    attn_map = attn.reshape(grid, grid).numpy()
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)

    # Upsample to original image size using PIL
    orig_w, orig_h = image.size
    attn_pil = Image.fromarray((attn_map * 255).astype(np.uint8)).resize(
        (orig_w, orig_h), Image.BICUBIC
    )
    # Light blur to smooth patch edges
    attn_pil = attn_pil.filter(ImageFilter.GaussianBlur(radius=orig_w // 30))

    # Build RGBA heat overlay (red channel = attention)
    attn_np = np.array(attn_pil, dtype=np.float32) / 255.0
    overlay_rgba = np.zeros((orig_h, orig_w, 4), dtype=np.uint8)
    overlay_rgba[..., 0] = (attn_np * 220).astype(np.uint8)          # R
    overlay_rgba[..., 1] = ((1 - attn_np) * 60).astype(np.uint8)     # G (slight teal in low regions)
    overlay_rgba[..., 2] = 30                                          # B constant
    overlay_rgba[..., 3] = (attn_np * 180).astype(np.uint8)           # A (alpha)

    overlay = Image.fromarray(overlay_rgba, mode="RGBA")
    base = image.convert("RGBA")
    combined = Image.alpha_composite(base, overlay).convert("RGB")
    return combined


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _get_yes_no_token_ids(processor) -> tuple[int, int]:
    """Return the single-token IDs for 'Yes' and 'No'."""
    tok = processor.tokenizer
    yes_ids = tok("Yes", add_special_tokens=False)["input_ids"]
    no_ids  = tok("No",  add_special_tokens=False)["input_ids"]
    return yes_ids[0], no_ids[0]


def run_inference(
    image: Image.Image,
    model,
    processor,
    category: str,
    threshold: float,
) -> tuple[str, float, str, float, float, bool]:
    """Run one forward pass for confidence + one generate for the full reply.

    Args:
        image: PIL image to inspect.
        model: Loaded LLaVA model.
        processor: Matching processor.
        category: MVTec product category (injected into the prompt).
        threshold: P(Yes) threshold for binary decision.

    Returns:
        label (str): "Anomaly detected" or "No anomaly detected"
        confidence (float): probability of the predicted class (0–1)
        description (str): full model-generated reply
        yes_pct (float): Yes-token probability as percentage (0–100)
        no_pct (float): No-token probability as percentage (0–100)
        is_anomaly (bool): binary decision
    """
    if image is None:
        return "—", 0.0, "Please upload an image.", 0.0, 0.0, False

    prompt = _make_prompt(category)

    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt",
    ).to(model.device)

    yes_id, no_id = _get_yes_no_token_ids(processor)

    # ── Confidence: single forward pass, look at next-token logits ──────────
    with torch.inference_mode():
        logits = model(**inputs).logits          # (1, seq_len, vocab)
        next_logits = logits[0, -1, :]           # logits at the last position

    yes_logit = next_logits[yes_id].float()
    no_logit  = next_logits[no_id].float()
    probs     = F.softmax(torch.stack([yes_logit, no_logit]), dim=0)
    yes_prob, no_prob = probs[0].item(), probs[1].item()

    is_anomaly = yes_prob > threshold
    confidence = yes_prob if is_anomaly else no_prob

    # ── Full reply: generate up to 64 new tokens ────────────────────────────
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
        )
    description = processor.tokenizer.decode(
        out[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True,
    ).strip()

    label = "Anomaly detected" if is_anomaly else "No anomaly detected"
    return label, round(confidence, 4), description, round(yes_prob * 100, 1), round(no_prob * 100, 1), is_anomaly


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def _find_example_images() -> list[list]:
    """Return example rows [image_path, category] from MVTec test set if present."""
    candidates = [
        ("mvtec_anomaly_detection/bottle/test/broken_large/000.png",  "bottle"),
        ("mvtec_anomaly_detection/bottle/test/contamination/000.png",  "bottle"),
        ("mvtec_anomaly_detection/bottle/test/good/000.png",           "bottle"),
        ("mvtec_anomaly_detection/cable/test/bent_wire/000.png",       "cable"),
        ("mvtec_anomaly_detection/screw/test/good/000.png",            "screw"),
        ("mvtec_anomaly_detection/leather/test/good/000.png",          "leather"),
    ]
    return [[p, cat] for p, cat in candidates if Path(p).exists()]


def build_interface(model, processor):
    import gradio as gr

    examples = _find_example_images()

    def predict(image, category, threshold):
        if image is None:
            return "Please upload an image.", "—", "—", None

        # Run binary classification
        label, confidence, description, yes_pct, no_pct, is_anomaly = run_inference(
            image, model, processor, category, threshold
        )

        # Build verdict markdown
        icon    = "🔴" if is_anomaly else "🟢"
        verdict = (
            f"{icon} **{label}**\n\n"
            f"- Confidence: **{confidence:.1%}**\n"
            f"- P(Yes): `{yes_pct:.1f}%`  |  P(No): `{no_pct:.1f}%`\n"
            f"- Threshold: `{threshold:.2f}` (category default: `{CATEGORY_THRESHOLDS.get(category, DEFAULT_THRESHOLD):.2f}`)"
        )

        # Compute attention overlay (may return None if unsupported)
        attn_img = compute_attention_overlay(image, model, processor, category)

        return verdict, description, attn_img

    with gr.Blocks(title="MVTec Defect Detection — LLaVA QLoRA") as demo:
        gr.Markdown(
            "# Manufacturing Defect Detection\n"
            "**LLaVA-1.5-7B** fine-tuned with **QLoRA** on the "
            "[MVTec Anomaly Detection](https://www.mvtec.com/company/research/datasets/mvtec-ad) "
            "dataset (15 industrial categories, v3 checkpoint).\n\n"
            "Upload an image, select the product category, and optionally adjust the "
            "detection threshold. The attention map shows *where* the vision encoder is looking."
        )

        with gr.Row():
            # ── Left column: inputs ─────────────────────────────────────────
            with gr.Column(scale=1):
                image_input = gr.Image(type="pil", label="Input image")

                category_dd = gr.Dropdown(
                    choices=CATEGORIES,
                    value="bottle",
                    label="Product category",
                    info="Select the MVTec category that matches your image. "
                         "The model was trained with category-aware prompts — "
                         "this must match the image type for best results.",
                )

                threshold_slider = gr.Slider(
                    minimum=0.05,
                    maximum=0.95,
                    step=0.05,
                    value=DEFAULT_THRESHOLD,
                    label="Detection threshold (P(Yes))",
                    info="Lower → more sensitive (higher recall, more false positives). "
                         "Higher → more conservative (fewer false positives, may miss defects). "
                         "Each category has a sweep-optimised default shown in the verdict.",
                )

                # Sync slider to per-category default when category changes
                def _update_threshold(cat):
                    return CATEGORY_THRESHOLDS.get(cat, DEFAULT_THRESHOLD)

                category_dd.change(
                    fn=_update_threshold,
                    inputs=category_dd,
                    outputs=threshold_slider,
                )

                submit_btn = gr.Button("Analyse", variant="primary")

            # ── Right column: outputs ────────────────────────────────────────
            with gr.Column(scale=1):
                verdict_out = gr.Markdown(label="Verdict")
                description_out = gr.Textbox(
                    label="Model description",
                    lines=3,
                    interactive=False,
                    info="Full text generated by the model.",
                )
                attn_out = gr.Image(
                    type="pil",
                    label="Attention map (CLIP vision encoder — last layer)",
                    interactive=False,
                )

        submit_btn.click(
            fn=predict,
            inputs=[image_input, category_dd, threshold_slider],
            outputs=[verdict_out, description_out, attn_out],
        )

        if examples:
            gr.Examples(
                examples=examples,
                inputs=[image_input, category_dd],
                label="MVTec test-set examples",
                examples_per_page=6,
            )
        else:
            gr.Markdown(
                "_Example images not found. Extract MVTec AD to "
                "`mvtec_anomaly_detection/` to enable built-in examples._"
            )

        gr.Markdown(
            "---\n"
            "**How it works:** The model uses "
            "[QLoRA](https://arxiv.org/abs/2305.14314) (4-bit NF4 quantization + "
            "LoRA adapters on attention + MLP layers, r=16) to fine-tune "
            "LLaVA-1.5-7B. Category-aware prompts, centre-crop augmentation for "
            "small-defect categories, and per-category detection thresholds are "
            "applied at inference time to match training conditions exactly.\n\n"
            "Global metrics on MVTec test split: **F1=0.834 · Recall=0.943 · ROC-AUC=0.896**"
        )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch Gradio defect-detection demo"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--repo-id",
        help="HuggingFace Hub repo ID of the merged model "
             "(e.g. your-username/llava-mvtec-defect-detection)",
    )
    source.add_argument(
        "--checkpoint",
        type=Path,
        help="Local LoRA checkpoint directory (requires --config too)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/local_8gb.yaml"),
        help="Training config YAML — only used with --checkpoint",
    )
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    if args.repo_id:
        model, processor = _load_from_hub(args.repo_id)
    else:
        model, processor = _load_from_checkpoint(args.checkpoint, args.config)

    demo = build_interface(model, processor)
    demo.launch(share=args.share, server_port=args.port)


if __name__ == "__main__":
    main()
