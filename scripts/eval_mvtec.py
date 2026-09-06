"""Evaluate MVTec VLM predictions: classification + rationale + grounding.

Inputs:
  --eval-file       held-out JSON in LLaVA format (see scripts/make_eval_split.py)
  --preds           model predictions JSON: list of {id, response} where
                    response is the raw model text; {id, label, rationale,
                    bbox_1000} is also accepted. Omit for a dataset-only report.
  --grounding-file  grounding subset JSON (see scripts/make_grounding_subset.py)
                    to score IoU@0.5 when preds carry bbox_1000 boxes.
  --mock            smoke-test mode: deterministic all-"No." dummy predictions
                    (pipeline check only — clearly labeled, never a result).
  --output          write full metrics JSON here (default: stdout only).

Metrics:
  * text accuracy overall + normal/defective splits (Yes/No parse)
  * per-class image-level AUROC (MVTec standard). Single-label classes
    report null — never a fabricated 0.5.
  * PRO: null by design — pixel-level PRO needs anomaly maps, which a
    text-only VLM does not emit. Box detection-rate @IoU>=0.5 on the
    grounding subset is reported as the proxy, labeled as such.
  * rationale factuality 0-3 (deterministic, no LLM judge):
    +1 defect-keyword hit, +1 single sentence, +1 no wrong-defect mention.

Usage:
    python scripts/eval_mvtec.py --eval-file eval_test.json --output eval_outputs/results.json
    python scripts/eval_mvtec.py --eval-file eval_test.json --preds eval_outputs/preds.json --output eval_outputs/results.json
    python scripts/eval_mvtec.py --eval-file eval_test.json --preds eval_outputs/preds.json --grounding-file mvtec_grounding_200.json
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

try:
    from sklearn.metrics import roc_auc_score
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

YES_RE = re.compile(r"\byes\b", re.IGNORECASE)
NO_RE = re.compile(r"\bno\b", re.IGNORECASE)
SENT_END_RE = re.compile(r"[.!?]")
BRACKET_RE = re.compile(r"\[.*?\]")
WORD_RE = re.compile(r"[a-z0-9]+")


def parse_label(text):
    """Return 'defective' / 'normal' / None from a Yes/No model response."""
    if text is None:
        return None
    head = str(text).strip().lower()[:10]
    if head.startswith("yes"):
        return "defective"
    if head.startswith("no"):
        return "normal"
    yes = YES_RE.search(str(text))
    no = NO_RE.search(str(text))
    if yes and not no:
        return "defective"
    if no and not yes:
        return "normal"
    if yes and no:  # first mention wins
        return "defective" if yes.start() < no.start() else "normal"
    return None


def ground_truth_label(item):
    """From the image path (/good/ -> normal), falling back to the reference answer."""
    parts = str(item.get("image", "")).split("/")
    if len(parts) >= 3:
        return "normal" if parts[2] == "good" else "defective"
    convs = item.get("conversations", [])
    if len(convs) >= 2:
        return "defective" if parse_label(convs[1].get("value", "")) == "defective" else "normal"
    return "normal"


def defect_of(item):
    parts = str(item.get("image", "")).split("/")
    if len(parts) >= 3 and parts[2] != "good":
        return parts[2]
    return None


def category_of(item):
    return str(item.get("image", "")).split("/")[0]


def normalize_pred(entry):
    """Accept {id, response|text|prediction} or {id, label, rationale, bbox_1000}."""
    pid = entry.get("id")
    text = entry.get("response", entry.get("text", entry.get("prediction")))
    if text is None:
        label = entry.get("label")
        text = str(label) if label is not None else ""
        if entry.get("rationale"):
            text = f"{text}. {entry['rationale']}"
    return pid, str(text), entry.get("bbox_1000")


def manual_auroc(labels, scores):
    pos = [s for l, s in zip(labels, scores) if l == 1]
    neg = [s for l, s in zip(labels, scores) if l == 0]
    if not pos or not neg:
        return None
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def per_class_auroc(label, scores):
    if len(set(label)) < 2:
        return None
    if _HAS_SKLEARN:
        try:
            return float(roc_auc_score(label, scores))
        except ValueError:
            return None
    return manual_auroc(label, scores)


def defect_vocab(items):
    vocab = set()
    for item in items:
        defect = defect_of(item)
        if defect:
            vocab.update(WORD_RE.findall(defect.lower()))
            vocab.add(defect.lower())
    return vocab


def rationale_scores(response, defect, vocab):
    """Deterministic 0-3 rubric. Returns dict with sub-scores."""
    text = BRACKET_RE.sub(" ", str(response or "")).lower()
    words = set(WORD_RE.findall(text))
    defect_tokens = set(WORD_RE.findall((defect or "").lower()))
    keyword_hit = bool(defect and defect_tokens and (defect_tokens & words))
    sentences = [s for s in SENT_END_RE.split(str(response or "").strip()) if s.strip()]
    single_sentence = len(sentences) <= 1
    others = {v for v in vocab if v not in defect_tokens} if defect else set(vocab)
    wrong_hit = any(
        len(tok) > 3 and tok in words
        for tok in others if "_" not in tok
    )
    score = int(keyword_hit) + int(single_sentence) + int(not wrong_hit)
    return {"keyword_hit": keyword_hit, "single_sentence": single_sentence,
            "no_wrong_defect": not wrong_hit, "score": score}


def box_iou(a, b):
    if not a or not b:
        return 0.0
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(description="Score MVTec VLM predictions.")
    parser.add_argument("--eval-file", default="eval_test.json")
    parser.add_argument("--preds", default=None)
    parser.add_argument("--grounding-file", default=None)
    parser.add_argument("--mock", action="store_true",
                        help="Smoke test with all-'No.' dummy predictions.")
    parser.add_argument("--output", default=None)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    args = parser.parse_args()

    with open(args.eval_file, encoding="utf-8") as f:
        eval_items = json.load(f)
    by_id = {e["id"]: e for e in eval_items}
    print(f"Loaded {len(eval_items)} eval items from {args.eval_file}")

    if args.preds:
        with open(args.preds, encoding="utf-8") as f:
            raw_preds = json.load(f)
        pred_map = {}
        for entry in raw_preds:
            pid, text, bbox = normalize_pred(entry)
            if pid in by_id:
                pred_map[pid] = (text, bbox)
        mock_mode = False
    elif args.mock:
        pred_map = {e["id"]: ("No.", None) for e in eval_items}
        mock_mode = True
        print("MOCK MODE: all-'No.' dummy predictions (pipeline check only).")
    else:
        pred_map = {}
        mock_mode = False

    vocab = defect_vocab(eval_items)
    rows, rationale_rows = [], []
    for item in eval_items:
        gt = ground_truth_label(item)
        cat = category_of(item)
        if item["id"] in pred_map:
            text, bbox = pred_map[item["id"]]
            pred = parse_label(text)
        else:
            text, bbox, pred = "", None, None
        rows.append({"category": cat, "gt": gt, "pred": pred,
                     "score": 1.0 if pred == "defective" else 0.0 if pred == "normal" else 0.5,
                     "parsed": pred is not None})
        defect = defect_of(item)
        if gt == "defective" and pred is not None:
            rationale_rows.append(rationale_scores(text, defect, vocab))

    summary = {
        "n_eval": len(eval_items),
        "n_scored": sum(1 for r in rows if r["parsed"]),
        "n_unparseable": sum(1 for r in rows if not r["parsed"]),
        "mock": mock_mode,
        "has_predictions": bool(pred_map),
        "text_accuracy": None, "normal_accuracy": None, "defective_accuracy": None,
        "mean_per_class_auroc": None,
        "pro": None,
        "pro_note": ("Pixel-level PRO needs anomaly maps, which a text-only VLM "
                     "does not emit; see grounding detection-rate @IoU>=0.5 as proxy."),
    }
    scored = [r for r in rows if r["parsed"]]
    if scored:
        summary["text_accuracy"] = sum(1 for r in scored if r["pred"] == r["gt"]) / len(scored)
        for split in ("normal", "defective"):
            group = [r for r in scored if r["gt"] == split]
            if group:
                summary[f"{split}_accuracy"] = sum(1 for r in group if r["pred"] == r["gt"]) / len(group)

    per_class = {}
    aurocs = []
    for cat in sorted({r["category"] for r in rows}):
        group = [r for r in rows if r["category"] == cat and r["parsed"]]
        entry = {"n": len(group), "accuracy": None, "auroc": None, "auroc_note": ""}
        if group:
            entry["accuracy"] = sum(1 for r in group if r["pred"] == r["gt"]) / len(group)
            labels = [1 if r["gt"] == "defective" else 0 for r in group]
            entry["auroc"] = per_class_auroc(labels, [r["score"] for r in group])
            if entry["auroc"] is None:
                entry["auroc_note"] = "single label present in predictions; AUROC undefined"
            else:
                aurocs.append(entry["auroc"])
        per_class[cat] = entry
    if aurocs:
        summary["mean_per_class_auroc"] = sum(aurocs) / len(aurocs)

    rationale = {"n": len(rationale_rows), "mean_score_0_3": None,
                 "keyword_hit_rate": None, "single_sentence_rate": None,
                 "no_wrong_defect_rate": None}
    if rationale_rows:
        rationale["mean_score_0_3"] = sum(r["score"] for r in rationale_rows) / len(rationale_rows)
        for key in ("keyword_hit", "single_sentence", "no_wrong_defect"):
            rationale[f"{key}_rate"] = sum(1 for r in rationale_rows if r[key]) / len(rationale_rows)

    grounding = {"n": 0, "mean_iou": None, "detection_rate_at_iou": None,
                 "iou_threshold": args.iou_threshold, "note": "no grounding preds scored"}
    if args.grounding_file and pred_map:
        with open(args.grounding_file, encoding="utf-8") as f:
            g_items = json.load(f)
        # Match preds to grounding items by id, falling back to image path
        # (grounding and eval splits carry independent uuids).
        img2pred = {by_id[pid]["image"]: pred for pid, pred in pred_map.items()
                    if pid in by_id}
        ious, hits = [], 0
        for g in g_items:
            pred_box = None
            if g.get("id") in pred_map:
                pred_box = pred_map[g["id"]][1]
            elif g.get("image") in img2pred:
                pred_box = img2pred[g["image"]][1]
            if pred_box and g.get("bbox_1000"):
                iou = box_iou(pred_box, g["bbox_1000"])
                ious.append(iou)
                hits += iou >= args.iou_threshold
        if ious:
            grounding = {"n": len(ious), "mean_iou": sum(ious) / len(ious),
                         "detection_rate_at_iou": hits / len(ious),
                         "iou_threshold": args.iou_threshold, "note": "box-level PRO proxy"}

    results = {"summary": summary, "per_class": per_class,
               "rationale": rationale, "grounding": grounding}

    print("\n| split | n | accuracy |")
    print("|---|---|---|")
    for split in ("text", "normal", "defective"):
        val = summary.get(f"{split}_accuracy")
        n = len(scored) if split == "text" else len([r for r in scored if r["gt"] == split])
        print(f"| {split} | {n} | {'—' if val is None else f'{val:.3f}'} |")
    print(f"\nmean per-class AUROC: {summary['mean_per_class_auroc'] if summary['mean_per_class_auroc'] is not None else '—'}")
    print(f"rationale mean (0-3): {rationale['mean_score_0_3'] if rationale['mean_score_0_3'] is not None else '—'} "
          f"(n={rationale['n']})")
    print(f"grounding: n={grounding['n']}, mean IoU={grounding['mean_iou']}, "
          f"det@{args.iou_threshold}={grounding['detection_rate_at_iou']}")
    if mock_mode:
        print("NOTE: mock predictions — numbers above validate the harness, not the model.")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved metrics to {args.output}")


if __name__ == "__main__":
    main()
