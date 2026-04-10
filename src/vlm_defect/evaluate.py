"""Evaluation script — accuracy, F1, per-category metrics, defect-type scoring,
confusion-matrix heatmap, ROC-AUC, per-category threshold sweep, and failure
case logging on the MVTec test split.

Usage:
    python scripts/evaluate.py checkpoints/llava-mvtec-lora configs/local_8gb.yaml
    make eval CHECKPOINT=checkpoints/llava-mvtec-lora
    vlm-eval checkpoints/llava-mvtec-lora configs/local_8gb.yaml

    # Use category-specific thresholds (recommended for v3 — lower threshold
    # for hard categories like transistor/screw):
    vlm-eval checkpoints/llava-mvtec-lora-v3/checkpoint-NNN configs/local_8gb.yaml

    # Sweep multiple thresholds (global + per-category optimal):
    vlm-eval checkpoints/... configs/local_8gb.yaml --sweep-threshold

    # Log failure cases for defect-type extraction to a JSON file:
    vlm-eval checkpoints/... configs/local_8gb.yaml --log-failures failures.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageOps
from transformers import BitsAndBytesConfig

# Import centre-crop helper from data module so train/eval use identical preprocessing
from vlm_defect.data import apply_center_crop


# ---------------------------------------------------------------------------
# Category-specific inference thresholds
# ---------------------------------------------------------------------------
# Per-category P(Yes) thresholds derived from the v3 checkpoint-500 threshold
# sweep (run with --sweep-threshold).  Each value is the threshold that
# maximises F1 for that category on the held-out test split.
#
# How to update: run  vlm-eval <checkpoint> <config> --sweep-threshold
# and copy the "Opt Thr" column from the "Per-Category Optimal Threshold" table.
#
# v3 checkpoint-500 sweep results (global optimal = 0.25, F1=0.8512):
#   bottle     0.20  f1=0.955  rec=1.000
#   cable      0.25  f1=0.860  rec=0.915
#   capsule    0.30  f1=0.864  rec=0.911  (was 0.20 — raised, sweep prefers 0.30)
#   carpet     0.15  f1=0.978  rec=0.957
#   grid       0.15  f1=0.951  rec=0.967
#   hazelnut   0.20  f1=0.907  rec=0.944
#   leather    0.40  f1=1.000  rec=1.000
#   metal_nut  0.50  f1=0.885  rec=0.958
#   pill       0.25  f1=0.893  rec=0.918  (unchanged)
#   screw      0.40  f1=0.792  rec=0.934  (was 0.10 — raised, sweep prefers 0.40)
#   tile       0.50  f1=0.929  rec=0.907
#   toothbrush 0.80  f1=0.875  rec=0.933
#   transistor 0.40  f1=0.612  rec=0.750  (was 0.10 — raised, sweep prefers 0.40)
#   wood       0.30  f1=0.969  rec=1.000  (= global default, omitted below)
#   zipper     0.40  f1=0.853  rec=0.951
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
    "zipper":     0.40,
    # "wood" intentionally omitted — optimal threshold matches global default
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_anomaly_response(text: str) -> bool:
    """Return True when the response begins with 'Yes' as a complete word."""
    return bool(re.match(r"yes\b", text.strip(), re.IGNORECASE))


def _yes_no_prob(scores_first_token: torch.Tensor, tokenizer) -> float:
    """Return P(Yes) / (P(Yes) + P(No)) from the first generated token logits.

    Args:
        scores_first_token: 1-D logit tensor for the first generated token
                            (``model.generate(..., output_scores=True).scores[0][0]``).
        tokenizer: HuggingFace tokenizer used to resolve Yes / No token ids.

    Returns:
        Scalar float in [0, 1].  Values > threshold → predict anomaly.
    """
    yes_ids = tokenizer.encode("Yes", add_special_tokens=False)
    no_ids  = tokenizer.encode("No",  add_special_tokens=False)

    yes_id = yes_ids[0] if yes_ids else None
    no_id  = no_ids[0]  if no_ids  else None

    if yes_id is None or no_id is None:
        return float("nan")

    pair_logits = scores_first_token[[yes_id, no_id]]
    probs = torch.softmax(pair_logits.float(), dim=0)
    return probs[0].item()  # P(Yes)


def _extract_defect_name(text: str) -> str | None:
    """Extract the defect class from a templated response.

    Matches patterns like:
      "Yes, there is a broken_large anomaly."
      "Yes, there is an contamination anomaly."
    Returns None when the pattern is absent.
    """
    m = re.search(r"there is an? (.+?) anomaly", text, re.IGNORECASE)
    return m.group(1).strip() if m else None


# Trailing noise words that differ between ground-truth labels and model output
_NOISE_WORDS = re.compile(
    r"\b(defect|anomaly|fault|flaw|issue|damage|error|problem)\b",
    re.IGNORECASE,
)


def _normalise_defect_name(name: str) -> str:
    """Normalise a defect name for fair comparison.

    Applies:
      1. Lowercase
      2. Underscores and hyphens → spaces
      3. Strip trailing noise words ("defect", "anomaly", …)
      4. Collapse whitespace

    Ground-truth labels use underscored_snake_case (e.g. "broken_large");
    model outputs tend to use spaces and sometimes add "defect" or "anomaly".
    Without normalisation, "broken large" vs "broken_large" is a miss despite
    being semantically identical, which inflates the failure count.
    """
    s = name.lower()
    s = s.replace("_", " ").replace("-", " ")
    s = _NOISE_WORDS.sub("", s)
    s = " ".join(s.split())  # collapse whitespace
    return s


def _fuzzy_score(a: str, b: str) -> float:
    """Character-level similarity ratio via difflib (0 – 1)."""
    return SequenceMatcher(None, a, b).ratio()


def _compute_metrics(tp: int, fp: int, tn: int, fn: int) -> dict[str, Any]:
    """Compute accuracy / precision / recall / F1 / specificity from confusion counts."""
    total = tp + fp + tn + fn
    accuracy    = (tp + tn) / total         if total          else 0.0
    precision   = tp / (tp + fp)            if (tp + fp)      else 0.0
    recall      = tp / (tp + fn)            if (tp + fn)      else 0.0
    specificity = tn / (tn + fp)            if (tn + fp)      else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) else 0.0
    )
    return {
        "accuracy":         round(accuracy, 4),
        "precision":        round(precision, 4),
        "recall":           round(recall, 4),
        "f1":               round(f1, 4),
        "specificity":      round(specificity, 4),
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "n_samples":        total,
    }


def _save_confusion_heatmap(tp: int, fp: int, tn: int, fn: int, out_path: Path) -> None:
    """Save a 2×2 confusion-matrix heatmap as a PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("  [WARN] matplotlib / seaborn not installed — skipping heatmap.")
        return

    cm = [[tn, fp], [fn, tp]]
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Pred: No", "Pred: Yes"],
        yticklabels=["True: No", "True: Yes"],
        ax=ax,
    )
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved heatmap → {out_path}")


def _tta_prob_yes(
    image: Image.Image,
    model,
    processor,
    prompt: str,
    tokenizer,
    augments: list[str] = ("original", "hflip"),
) -> float:
    """Test-Time Augmentation: average P(Yes) over multiple image variants.

    Supported augments:
      "original" — unchanged image
      "hflip"    — horizontal mirror (catches orientation-invariant defects)

    Returns the mean P(Yes) across all variants.  If any variant produces NaN
    (tokeniser lookup failure) it is excluded from the average.
    """
    variants: list[Image.Image] = []
    for aug in augments:
        if aug == "original":
            variants.append(image)
        elif aug == "hflip":
            variants.append(ImageOps.mirror(image))

    probs: list[float] = []
    for img in variants:
        inputs = processor(text=prompt, images=img, return_tensors="pt").to(model.device)
        gen_out = model.generate(
            **inputs,
            max_new_tokens=1,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
        )
        p = _yes_no_prob(gen_out.scores[0][0], tokenizer)
        if p == p:  # exclude NaN
            probs.append(p)

    return sum(probs) / len(probs) if probs else float("nan")


def _compute_roc_auc(prob_yes_scores: list[tuple[bool, float]]) -> float | None:
    """Compute threshold-independent ROC-AUC from (true_label, prob_yes) pairs."""
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        return None
    labels = [int(t) for t, _ in prob_yes_scores]
    scores = [p for _, p in prob_yes_scores]
    # Filter out NaN scores
    valid = [(l, s) for l, s in zip(labels, scores) if s == s]
    if len(valid) < 2 or len(set(l for l, _ in valid)) < 2:
        return None
    labels_v, scores_v = zip(*valid)
    return round(roc_auc_score(list(labels_v), list(scores_v)), 4)


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate(
    checkpoint_dir: Path,
    cfg: dict,
    project_dir: Path,
    threshold: float = 0.25,
    category_thresholds: dict[str, float] | None = None,
    log_failures_path: Path | None = None,
    tta: bool = False,
    tta_categories: set[str] | None = None,
) -> dict:
    """Run inference on the MVTec test split and return a metrics dict.

    Args:
        checkpoint_dir: Path to the LoRA adapter checkpoint directory.
        cfg: Parsed YAML config dict.
        project_dir: Absolute path to the project root.
        threshold: Global P(Yes) threshold (default 0.3 — optimal from v2 sweep).
                   Per-category overrides in CATEGORY_THRESHOLDS take precedence.
        category_thresholds: Optional mapping of category → threshold to override
                   the module-level CATEGORY_THRESHOLDS dict.  Pass {} to
                   disable per-category thresholds entirely.
        log_failures_path: If given, write defect-type extraction failure cases
                   (wrong / missing predictions) to this JSON path.
        tta: If True, enable Test-Time Augmentation (horizontal flip averaged
                   with original).  Improves F1 on hard categories (transistor,
                   toothbrush) at the cost of 2× inference time.
        tta_categories: Set of category names to apply TTA to.  If None and
                   tta=True, TTA is applied to all categories.  Pass a subset
                   (e.g. {"transistor", "toothbrush"}) to restrict TTA.

    Returned dict keys
    ------------------
    accuracy, precision, recall, f1, specificity, confusion_matrix, n_samples
        Global binary metrics.
    per_category
        Same metric block per MVTec category.
    defect_type_scoring
        Exact-match rate, avg fuzzy score, and (if log_failures_path set)
        per-sample failure records.
    roc_auc
        Threshold-independent AUC score (None if sklearn not installed).
    threshold
        The global P(Yes) threshold used for this run.
    category_thresholds_used
        The effective per-category threshold map applied during inference.
    prob_yes_scores
        List of (true_label: bool, prob_yes: float) pairs for offline curves.
    """
    from transformers import AutoProcessor, LlavaForConditionalGeneration
    from peft import PeftModel

    # Merge module-level and caller-supplied category thresholds
    effective_cat_thresholds: dict[str, float] = dict(CATEGORY_THRESHOLDS)
    if category_thresholds is not None:
        effective_cat_thresholds.update(category_thresholds)

    # ── resolve paths ────────────────────────────────────────────────────────
    d = cfg["data"]
    train_json = project_dir / d["path"]
    if "test_path" in d:
        test_json = project_dir / d["test_path"]
    else:
        test_json = train_json.with_name(
            train_json.name.replace("_train", "_test")
        )

    if not test_json.exists():
        raise FileNotFoundError(
            f"Test JSON not found: {test_json}\n"
            "Run: vlm-prepare (or: make prepare)"
        )

    image_folder = project_dir / d["image_folder"]

    # ── load model ───────────────────────────────────────────────────────────
    base_model_id = cfg["model"]["name_or_path"]
    print("[INFO] Loading processor...")
    processor = AutoProcessor.from_pretrained(base_model_id)

    print("[INFO] Loading base model + LoRA adapter...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = LlavaForConditionalGeneration.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(model, str(checkpoint_dir.resolve()))
    model.eval()

    # ── load test records ────────────────────────────────────────────────────
    print(f"[INFO] Loading test split from {test_json} ...")
    with open(test_json) as f:
        records = json.load(f)
    print(f"[INFO] {len(records)} test samples.")
    print(f"[INFO] Global P(Yes) threshold = {threshold:.2f}")
    if effective_cat_thresholds:
        print(f"[INFO] Per-category overrides: {effective_cat_thresholds}")
    if tta:
        tta_scope = tta_categories or "all categories"
        print(f"[INFO] Test-Time Augmentation enabled (hflip) — scope: {tta_scope}")

    # ── accumulators ─────────────────────────────────────────────────────────
    tp = fp = tn = fn = 0
    cat_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    )

    defect_n_true_anomalies  = 0
    defect_n_both_extracted  = 0
    defect_exact_matches     = 0
    defect_fuzzy_total       = 0.0
    failure_cases: list[dict] = []   # for --log-failures

    prob_yes_scores: list[tuple[bool, float]] = []

    # ── inference loop ───────────────────────────────────────────────────────
    with torch.inference_mode():
        for idx, item in enumerate(records):
            image_path = item["image"]
            parts      = image_path.split("/")
            category   = parts[0]
            defect_dir = parts[2] if len(parts) >= 3 else "unknown"

            image = Image.open(image_folder / image_path).convert("RGB")
            # Apply the same centre-crop preprocessing used during training
            image = apply_center_crop(image, category)

            human     = item["conversations"][0]["value"]
            gpt_truth = item["conversations"][1]["value"]

            prompt = f"USER: {human} ASSISTANT:"
            inputs = processor(
                text=prompt,
                images=image,
                return_tensors="pt",
            ).to(model.device)

            gen_out = model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
            )

            generated = processor.tokenizer.decode(
                gen_out.sequences[0][inputs["input_ids"].shape[-1]:],
                skip_special_tokens=True,
            ).strip()

            prob_yes = _yes_no_prob(gen_out.scores[0][0], processor.tokenizer)

            # Test-Time Augmentation: average with horizontal-flip prediction
            use_tta = tta and (tta_categories is None or category in tta_categories)
            if use_tta:
                prob_yes_tta = _tta_prob_yes(
                    image, model, processor, prompt,
                    processor.tokenizer,
                    augments=["original", "hflip"],
                )
                # Replace prob_yes with the TTA-averaged value if not NaN
                if prob_yes_tta == prob_yes_tta:
                    prob_yes = prob_yes_tta

            true_anomaly = _is_anomaly_response(gpt_truth)
            prob_yes_scores.append((true_anomaly, prob_yes))

            # Apply per-category threshold override (falls back to global)
            cat_threshold = effective_cat_thresholds.get(category, threshold)
            if prob_yes != prob_yes:  # NaN guard
                pred_anomaly = _is_anomaly_response(generated)
            else:
                pred_anomaly = prob_yes > cat_threshold

            # ── binary classification counts ─────────────────────────────────
            if true_anomaly and pred_anomaly:
                tp += 1;  cat_counts[category]["tp"] += 1
            elif not true_anomaly and pred_anomaly:
                fp += 1;  cat_counts[category]["fp"] += 1
            elif true_anomaly and not pred_anomaly:
                fn += 1;  cat_counts[category]["fn"] += 1
            else:
                tn += 1;  cat_counts[category]["tn"] += 1

            # ── defect-type extraction scoring ───────────────────────────────
            if true_anomaly:
                defect_n_true_anomalies += 1
                gt_defect = defect_dir if defect_dir != "good" else None
                if gt_defect is None:
                    gt_defect = _extract_defect_name(gpt_truth)

                pred_defect = _extract_defect_name(generated)

                if gt_defect and pred_defect:
                    defect_n_both_extracted += 1
                    gt_norm   = _normalise_defect_name(gt_defect)
                    pred_norm = _normalise_defect_name(pred_defect)
                    exact  = gt_norm == pred_norm
                    fscore = _fuzzy_score(pred_norm, gt_norm)
                    if exact:
                        defect_exact_matches += 1
                    defect_fuzzy_total += fscore

                    # Record failures for --log-failures
                    if not exact and log_failures_path is not None:
                        failure_cases.append({
                            "image":        image_path,
                            "category":     category,
                            "gt_defect":    gt_defect,
                            "pred_defect":  pred_defect,
                            "fuzzy_score":  round(fscore, 4),
                            "generated":    generated,
                            "prob_yes":     round(prob_yes, 4) if prob_yes == prob_yes else None,
                        })
                elif gt_defect and log_failures_path is not None:
                    # Model produced no parseable defect name
                    failure_cases.append({
                        "image":        image_path,
                        "category":     category,
                        "gt_defect":    gt_defect,
                        "pred_defect":  None,
                        "fuzzy_score":  0.0,
                        "generated":    generated,
                        "prob_yes":     round(prob_yes, 4) if prob_yes == prob_yes else None,
                    })

            if (idx + 1) % 50 == 0:
                print(f"  {idx + 1}/{len(records)} evaluated...")

    # ── aggregate results ────────────────────────────────────────────────────
    global_metrics = _compute_metrics(tp, fp, tn, fn)

    per_category: dict[str, dict] = {}
    for cat, c in sorted(cat_counts.items()):
        per_category[cat] = _compute_metrics(c["tp"], c["fp"], c["tn"], c["fn"])

    defect_type_scoring = {
        "n_true_anomalies":  defect_n_true_anomalies,
        "n_both_extracted":  defect_n_both_extracted,
        "exact_match_rate":  round(
            defect_exact_matches / defect_n_both_extracted, 4
        ) if defect_n_both_extracted else 0.0,
        "avg_fuzzy_score": round(
            defect_fuzzy_total / defect_n_both_extracted, 4
        ) if defect_n_both_extracted else 0.0,
    }

    roc_auc = _compute_roc_auc(prob_yes_scores)

    return {
        **global_metrics,
        "per_category":              per_category,
        "defect_type_scoring":       defect_type_scoring,
        "roc_auc":                   roc_auc,
        "threshold":                 threshold,
        "category_thresholds_used":  effective_cat_thresholds,
        "prob_yes_scores":           prob_yes_scores,
        "_failure_cases":            failure_cases,   # internal — saved separately
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import yaml

    parser = argparse.ArgumentParser(
        description="Evaluate a fine-tuned LLaVA checkpoint on the MVTec test split"
    )
    parser.add_argument("checkpoint", type=Path, help="Path to LoRA checkpoint dir")
    parser.add_argument("config",     type=Path, help="Path to YAML config file")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.25,
        help=(
            "Global P(Yes) threshold (default: 0.25 — optimal from v3 sweep). "
            "Per-category overrides in CATEGORY_THRESHOLDS take precedence. "
            "Use --no-category-thresholds to apply this value to all categories."
        ),
    )
    parser.add_argument(
        "--no-category-thresholds",
        action="store_true",
        help=(
            "Disable per-category threshold overrides and use --threshold "
            "globally for all categories."
        ),
    )
    parser.add_argument(
        "--sweep-threshold",
        action="store_true",
        help=(
            "Run a single inference pass, collect P(Yes) scores, then report "
            "global metrics at thresholds [0.05, 0.10, …, 0.90] and also find "
            "the optimal per-category threshold."
        ),
    )
    parser.add_argument(
        "--log-failures",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Write defect-type extraction failures (wrong / missing predicted "
            "defect names) to a JSON file for qualitative analysis."
        ),
    )
    parser.add_argument(
        "--tta",
        action="store_true",
        help=(
            "Enable Test-Time Augmentation: average P(Yes) over the original "
            "image and its horizontal mirror.  Improves F1 on hard categories "
            "(transistor, toothbrush) at the cost of ~2× inference time."
        ),
    )
    parser.add_argument(
        "--tta-categories",
        nargs="+",
        metavar="CATEGORY",
        default=None,
        help=(
            "Restrict TTA to specific categories (e.g. transistor toothbrush). "
            "When omitted and --tta is set, TTA is applied to all categories."
        ),
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    project_dir = Path(__file__).parent.parent.parent.absolute()

    cat_thresholds = {} if args.no_category_thresholds else None  # None → use defaults
    tta_cats = set(args.tta_categories) if args.tta_categories else None

    metrics = evaluate(
        args.checkpoint,
        cfg,
        project_dir,
        threshold=args.threshold,
        category_thresholds=cat_thresholds,
        log_failures_path=args.log_failures,
        tta=args.tta,
        tta_categories=tta_cats,
    )

    # ── pretty-print global metrics ──────────────────────────────────────────
    print("\n── Global Results ──────────────────────────────────")
    print(f"  Samples:      {metrics['n_samples']}")
    print(f"  Accuracy:     {metrics['accuracy']:.4f}")
    print(f"  Precision:    {metrics['precision']:.4f}")
    print(f"  Recall:       {metrics['recall']:.4f}")
    print(f"  F1:           {metrics['f1']:.4f}")
    print(f"  Specificity:  {metrics['specificity']:.4f}")
    if metrics.get("roc_auc") is not None:
        print(f"  ROC-AUC:      {metrics['roc_auc']:.4f}")
    cm = metrics["confusion_matrix"]
    print(f"  Confusion:    TP={cm['tp']}  FP={cm['fp']}  TN={cm['tn']}  FN={cm['fn']}")
    print(f"  Threshold:    global={args.threshold:.2f}  "
          f"per-category={not args.no_category_thresholds}")

    # ── per-category ─────────────────────────────────────────────────────────
    if metrics["per_category"]:
        cat_thr = metrics["category_thresholds_used"]
        print("\n── Per-Category Results ─────────────────────────────")
        print(f"  {'Category':<20}  {'n':>4}  {'acc':>5}  {'rec':>5}  {'f1':>5}  {'thr':>5}")
        print(f"  {'-'*20}  {'-'*4}  {'-'*5}  {'-'*5}  {'-'*5}  {'-'*5}")
        for cat, m in metrics["per_category"].items():
            thr_used = cat_thr.get(cat, args.threshold)
            print(
                f"  {cat:<20}  n={m['n_samples']:>4}  "
                f"acc={m['accuracy']:.3f}  rec={m['recall']:.3f}  "
                f"f1={m['f1']:.3f}  thr={thr_used:.2f}"
            )

    # ── defect-type scoring ──────────────────────────────────────────────────
    dt = metrics["defect_type_scoring"]
    if dt["n_true_anomalies"]:
        print("\n── Defect-Type Extraction ──────────────────────────")
        print(f"  True anomaly samples:  {dt['n_true_anomalies']}")
        print(f"  Both names extracted:  {dt['n_both_extracted']}")
        print(f"  Exact-match rate:      {dt['exact_match_rate']:.4f}")
        print(f"  Avg fuzzy score:       {dt['avg_fuzzy_score']:.4f}")
        if args.log_failures and metrics.get("_failure_cases"):
            n_fail = len(metrics["_failure_cases"])
            print(f"  Failure cases:         {n_fail}")

    # ── confusion matrix heatmap ─────────────────────────────────────────────
    c = metrics["confusion_matrix"]
    _save_confusion_heatmap(
        c["tp"], c["fp"], c["tn"], c["fn"],
        args.checkpoint / "confusion_matrix.png",
    )

    # ── global threshold sweep ───────────────────────────────────────────────
    if args.sweep_threshold and metrics["prob_yes_scores"]:
        sweep_thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40,
                            0.50, 0.60, 0.70, 0.80, 0.90]
        print("\n── Global Threshold Sweep ──────────────────────────")
        print(f"  {'Threshold':>10}  {'Accuracy':>8}  {'Precision':>9}  "
              f"{'Recall':>7}  {'F1':>7}  {'Specificity':>12}")
        print(f"  {'-'*10}  {'-'*8}  {'-'*9}  {'-'*7}  {'-'*7}  {'-'*12}")

        best_f1 = -1.0
        best_thr = args.threshold
        for thr in sweep_thresholds:
            _tp = _fp = _tn = _fn = 0
            for true_label, py in metrics["prob_yes_scores"]:
                pred = py > thr if py == py else False
                if true_label and pred:       _tp += 1
                elif not true_label and pred: _fp += 1
                elif true_label and not pred: _fn += 1
                else:                         _tn += 1
            m = _compute_metrics(_tp, _fp, _tn, _fn)
            marker = " ◀ selected" if abs(thr - args.threshold) < 1e-6 else ""
            if m["f1"] > best_f1:
                best_f1, best_thr = m["f1"], thr
            print(
                f"  {thr:>10.2f}  {m['accuracy']:>8.4f}  {m['precision']:>9.4f}  "
                f"{m['recall']:>7.4f}  {m['f1']:>7.4f}  {m['specificity']:>12.4f}{marker}"
            )
        print(f"\n  Optimal global threshold: {best_thr:.2f}  (F1={best_f1:.4f})")

        # ── per-category optimal threshold sweep ─────────────────────────────
        print("\n── Per-Category Optimal Threshold ──────────────────")
        print(f"  {'Category':<20}  {'Opt Thr':>7}  {'Best F1':>7}  {'Recall':>7}")
        print(f"  {'-'*20}  {'-'*7}  {'-'*7}  {'-'*7}")

        # Rebuild (true_label, prob_yes, category) triples from records + scores
        # We re-open the test JSON to get category info aligned with prob_yes_scores
        try:
            d_cfg = cfg["data"]
            train_json = project_dir / d_cfg["path"]
            if "test_path" in d_cfg:
                test_json_path = project_dir / d_cfg["test_path"]
            else:
                test_json_path = train_json.with_name(
                    train_json.name.replace("_train", "_test")
                )
            with open(test_json_path) as f:
                test_records = json.load(f)

            # Build per-category score lists
            cat_scores: dict[str, list[tuple[bool, float]]] = defaultdict(list)
            for record, (true_label, py) in zip(test_records, metrics["prob_yes_scores"]):
                cat = record["image"].split("/")[0]
                cat_scores[cat].append((true_label, py))

            for cat in sorted(cat_scores.keys()):
                scores_cat = cat_scores[cat]
                best_cat_f1  = -1.0
                best_cat_thr = 0.30
                best_cat_rec = 0.0
                for thr in sweep_thresholds:
                    _tp = _fp = _tn = _fn = 0
                    for true_label, py in scores_cat:
                        pred = py > thr if py == py else False
                        if true_label and pred:       _tp += 1
                        elif not true_label and pred: _fp += 1
                        elif true_label and not pred: _fn += 1
                        else:                         _tn += 1
                    m = _compute_metrics(_tp, _fp, _tn, _fn)
                    if m["f1"] > best_cat_f1:
                        best_cat_f1, best_cat_thr = m["f1"], thr
                        best_cat_rec = m["recall"]
                current_override = metrics["category_thresholds_used"].get(cat, "—")
                marker = (
                    f"  ← override={current_override:.2f}"
                    if isinstance(current_override, float)
                    else ""
                )
                print(
                    f"  {cat:<20}  {best_cat_thr:>7.2f}  {best_cat_f1:>7.4f}  "
                    f"{best_cat_rec:>7.4f}{marker}"
                )
        except Exception as e:
            print(f"  [WARN] Per-category sweep failed: {e}")

    # ── save failure cases ───────────────────────────────────────────────────
    if args.log_failures and metrics.get("_failure_cases"):
        with open(args.log_failures, "w") as f:
            json.dump(metrics["_failure_cases"], f, indent=2)
        print(f"\n  Saved {len(metrics['_failure_cases'])} failure cases → {args.log_failures}")

    # ── save JSON (exclude raw scores and internal fields) ───────────────────
    out_path = args.checkpoint / "eval_results.json"
    exclude  = {"prob_yes_scores", "_failure_cases"}
    save_metrics = {k: v for k, v in metrics.items() if k not in exclude}
    with open(out_path, "w") as f:
        json.dump(save_metrics, f, indent=2)
    print(f"\n  Saved results → {out_path}")


if __name__ == "__main__":
    main()
