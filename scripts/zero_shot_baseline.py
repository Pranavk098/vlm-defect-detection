#!/usr/bin/env python3
"""Zero-shot baseline evaluation — base LLaVA-1.5-7B without any LoRA adapter.

Runs the same evaluation pipeline as vlm-eval but loads the raw base model,
establishing the performance floor before fine-tuning.  Compare against the
v3 checkpoint to quantify the improvement from QLoRA training.

Usage:
    python scripts/zero_shot_baseline.py configs/local_8gb.yaml

    # Save results to a JSON file for compare_checkpoints.py:
    python scripts/zero_shot_baseline.py configs/local_8gb.yaml \\
        --out-dir checkpoints/zero_shot_baseline

    # Use per-category thresholds (same as fine-tuned eval):
    python scripts/zero_shot_baseline.py configs/local_8gb.yaml \\
        --sweep-threshold

Expected baseline (approximate, base LLaVA-1.5-7B):
    The base model answers most questions generically ("I see a bottle") without
    specialising on anomaly detection.  Expected F1 ≈ 0.45–0.60, primarily from
    random-chance recall on an imbalanced test set.  The fine-tuned v3 model
    achieves F1=0.887, ROC-AUC=0.896 — document the delta in README.md.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
import yaml
from PIL import Image
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    LlavaForConditionalGeneration,
)

from vlm_defect.data import apply_center_crop
from vlm_defect.evaluate import (
    CATEGORY_THRESHOLDS,
    _compute_metrics,
    _compute_roc_auc,
    _is_anomaly_response,
    _save_confusion_heatmap,
    _yes_no_prob,
)


def _make_prompt(category: str) -> str:
    """Category-aware prompt — identical to fine-tuned model's training prompt."""
    return (
        f"USER: <image>\n"
        f"Is there any anomaly in this {category} image? "
        f"If yes, say 'Yes, there is a <defect_type> anomaly.' "
        f"If no, say 'No.' ASSISTANT:"
    )


def run_zero_shot(
    cfg: dict,
    project_dir: Path,
    threshold: float = 0.25,
    use_category_thresholds: bool = True,
) -> dict:
    """Evaluate the base model (no adapter) on the MVTec test split.

    Uses the same preprocessing, prompts, and threshold logic as evaluate.py
    so results are directly comparable to the fine-tuned checkpoint.
    """
    base_model_id = cfg["model"]["name_or_path"]
    d = cfg["data"]
    train_json = project_dir / d["path"]
    test_json  = train_json.with_name(train_json.name.replace("_train", "_test"))
    if not test_json.exists():
        raise FileNotFoundError(
            f"Test JSON not found: {test_json}\nRun: make prepare"
        )
    image_folder = project_dir / d["image_folder"]

    # Load base model in 4-bit (same memory budget as fine-tuned eval)
    print(f"[INFO] Loading BASE model {base_model_id} (no LoRA adapter) ...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    processor = AutoProcessor.from_pretrained(base_model_id)
    model = LlavaForConditionalGeneration.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
    )
    model.eval()

    print("[INFO] No LoRA adapter loaded — this is the zero-shot baseline.")
    print(f"[INFO] Threshold = {threshold:.2f}, per-category = {use_category_thresholds}")

    with open(test_json) as f:
        records = json.load(f)
    print(f"[INFO] {len(records)} test samples.")

    tp = fp = tn = fn = 0
    cat_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    )
    prob_yes_scores: list[tuple[bool, float]] = []

    with torch.inference_mode():
        for idx, item in enumerate(records):
            image_path = item["image"]
            category   = image_path.split("/")[0]

            image = Image.open(image_folder / image_path).convert("RGB")
            image = apply_center_crop(image, category)

            human     = item["conversations"][0]["value"]
            gpt_truth = item["conversations"][1]["value"]
            prompt    = f"USER: {human} ASSISTANT:"

            inputs = processor(
                text=prompt, images=image, return_tensors="pt"
            ).to(model.device)

            gen_out = model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
            )

            prob_yes    = _yes_no_prob(gen_out.scores[0][0], processor.tokenizer)
            true_anomaly = _is_anomaly_response(gpt_truth)
            prob_yes_scores.append((true_anomaly, prob_yes))

            cat_thr = CATEGORY_THRESHOLDS.get(category, threshold) if use_category_thresholds else threshold
            if prob_yes != prob_yes:  # NaN guard
                generated = processor.tokenizer.decode(
                    gen_out.sequences[0][inputs["input_ids"].shape[-1]:],
                    skip_special_tokens=True,
                ).strip()
                pred_anomaly = _is_anomaly_response(generated)
            else:
                pred_anomaly = prob_yes > cat_thr

            if true_anomaly and pred_anomaly:
                tp += 1
                cat_counts[category]["tp"] += 1
            elif not true_anomaly and pred_anomaly:
                fp += 1
                cat_counts[category]["fp"] += 1
            elif true_anomaly and not pred_anomaly:
                fn += 1
                cat_counts[category]["fn"] += 1
            else:
                tn += 1
                cat_counts[category]["tn"] += 1

            if (idx + 1) % 50 == 0:
                print(f"  {idx + 1}/{len(records)} evaluated...")

    global_metrics = _compute_metrics(tp, fp, tn, fn)
    per_category = {
        cat: _compute_metrics(c["tp"], c["fp"], c["tn"], c["fn"])
        for cat, c in sorted(cat_counts.items())
    }
    roc_auc = _compute_roc_auc(prob_yes_scores)

    return {
        **global_metrics,
        "per_category": per_category,
        "roc_auc": roc_auc,
        "threshold": threshold,
        "model": base_model_id,
        "lora": "none (zero-shot baseline)",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Zero-shot baseline: evaluate base LLaVA without LoRA adapter"
    )
    parser.add_argument("config", type=Path, help="Path to YAML training config")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.25,
        help="Global P(Yes) threshold (default: 0.25, same as fine-tuned eval)",
    )
    parser.add_argument(
        "--no-category-thresholds",
        action="store_true",
        help="Disable per-category threshold overrides",
    )
    parser.add_argument(
        "--sweep-threshold",
        action="store_true",
        help="Print metrics at thresholds 0.05–0.90 after inference",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory to save eval_results.json and confusion_matrix.png",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    project_dir = Path(__file__).parent.parent.absolute()

    metrics = run_zero_shot(
        cfg,
        project_dir,
        threshold=args.threshold,
        use_category_thresholds=not args.no_category_thresholds,
    )

    # ── Print global metrics ─────────────────────────────────────────────────
    print("\n── Zero-Shot Baseline Results ──────────────────────")
    print(f"  Model:        {metrics['model']} (no fine-tuning)")
    print(f"  Samples:      {metrics['n_samples']}")
    print(f"  Accuracy:     {metrics['accuracy']:.4f}")
    print(f"  Precision:    {metrics['precision']:.4f}")
    print(f"  Recall:       {metrics['recall']:.4f}")
    print(f"  F1:           {metrics['f1']:.4f}")
    print(f"  Specificity:  {metrics['specificity']:.4f}")
    if metrics.get("roc_auc"):
        print(f"  ROC-AUC:      {metrics['roc_auc']:.4f}")
    cm = metrics["confusion_matrix"]
    print(f"  Confusion:    TP={cm['tp']}  FP={cm['fp']}  TN={cm['tn']}  FN={cm['fn']}")

    print("\n── Comparison ──────────────────────────────────────")
    print(f"  {'Metric':<12}  {'Zero-shot':>10}  {'v3 fine-tuned':>14}  {'Delta':>8}")
    print(f"  {'-'*12}  {'-'*10}  {'-'*14}  {'-'*8}")
    v3 = {"f1": 0.8865, "recall": 0.9398, "precision": 0.8388,
          "accuracy": 0.8601, "roc_auc": 0.8960}
    for met in ["f1", "recall", "precision", "accuracy", "roc_auc"]:
        zs = metrics.get(met, 0) or 0
        ft = v3[met]
        delta = ft - zs
        print(f"  {met:<12}  {zs:>10.4f}  {ft:>14.4f}  {delta:>+8.4f}")

    # ── Per-category ─────────────────────────────────────────────────────────
    if metrics.get("per_category"):
        print("\n── Per-Category ────────────────────────────────────")
        print(f"  {'Category':<20}  {'n':>4}  {'acc':>6}  {'rec':>6}  {'f1':>6}")
        print(f"  {'-'*20}  {'-'*4}  {'-'*6}  {'-'*6}  {'-'*6}")
        for cat, m in metrics["per_category"].items():
            print(
                f"  {cat:<20}  {m['n_samples']:>4}  "
                f"{m['accuracy']:.3f}  {m['recall']:.3f}  {m['f1']:.3f}"
            )

    # ── Threshold sweep ──────────────────────────────────────────────────────
    if args.sweep_threshold and metrics.get("prob_yes_scores"):
        from vlm_defect.evaluate import _compute_metrics as _cm
        thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70]
        print("\n── Global Threshold Sweep ──────────────────────────")
        print(f"  {'Threshold':>10}  {'F1':>7}  {'Recall':>7}  {'Precision':>9}")
        best_f1, best_thr = -1.0, 0.25
        for thr in thresholds:
            _tp = _fp = _tn = _fn = 0
            for true_label, py in metrics["prob_yes_scores"]:
                pred = py > thr if py == py else False
                if true_label and pred:
                    _tp += 1
                elif not true_label and pred:
                    _fp += 1
                elif true_label and not pred:
                    _fn += 1
                else:
                    _tn += 1
            m = _cm(_tp, _fp, _tn, _fn)
            if m["f1"] > best_f1:
                best_f1, best_thr = m["f1"], thr
            print(f"  {thr:>10.2f}  {m['f1']:>7.4f}  {m['recall']:>7.4f}  {m['precision']:>9.4f}")
        print(f"\n  Optimal threshold: {best_thr:.2f}  (F1={best_f1:.4f})")

    # ── Save results ─────────────────────────────────────────────────────────
    if args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        out_path = args.out_dir / "eval_results.json"
        save = {k: v for k, v in metrics.items() if k != "prob_yes_scores"}
        out_path.write_text(json.dumps(save, indent=2))
        print(f"\n  Saved results → {out_path}")

        c = metrics["confusion_matrix"]
        _save_confusion_heatmap(
            c["tp"], c["fp"], c["tn"], c["fn"],
            args.out_dir / "confusion_matrix.png",
        )


if __name__ == "__main__":
    main()
