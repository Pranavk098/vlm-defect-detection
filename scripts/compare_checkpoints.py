#!/usr/bin/env python3
"""Compare evaluation results across checkpoints and across training versions.

Reads every ``checkpoint-<step>/eval_results.json`` under one or more checkpoint
directories, prints a consolidated table sorted by step (or a chosen metric),
and highlights the best checkpoint by F1.  Optionally prints per-category F1 and
defect-type extraction stats.

Use this to decide which checkpoint to push to HuggingFace Hub before running
push_to_hub.py, and to document the v1 → v2 → v3 improvement story.

Usage:
    # Single version
    python scripts/compare_checkpoints.py checkpoints/llava-mvtec-lora-v3
    python scripts/compare_checkpoints.py checkpoints/llava-mvtec-lora-v3 --sort f1
    python scripts/compare_checkpoints.py checkpoints/llava-mvtec-lora-v3 --per-category

    # Compare multiple versions side-by-side (best checkpoint from each)
    python scripts/compare_checkpoints.py \\
        checkpoints/llava-mvtec-lora \\
        checkpoints/llava-mvtec-lora-v2 \\
        checkpoints/llava-mvtec-lora-v3 \\
        --compare-versions

    # Full analysis: sort by F1, per-category breakdown, defect extraction stats
    python scripts/compare_checkpoints.py checkpoints/llava-mvtec-lora-v3 \\
        --sort f1 --per-category --defect-stats
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


METRICS = ["accuracy", "precision", "recall", "f1"]
COLUMN_WIDTHS = {
    "step":      6,
    "n_samples": 9,
    "accuracy":  8,
    "precision": 9,
    "recall":    6,
    "f1":        6,
    "roc_auc":   7,
}


def _step_from_name(name: str) -> int:
    """Extract the step number from a checkpoint directory name."""
    m = re.search(r"(\d+)$", name)
    return int(m.group(1)) if m else -1


def _load_results(checkpoint_root: Path) -> list[dict]:
    """Scan *checkpoint_root* for sub-dirs with eval_results.json."""
    rows = []
    for subdir in sorted(checkpoint_root.iterdir()):
        if not subdir.is_dir():
            continue
        result_file = subdir / "eval_results.json"
        if not result_file.exists():
            continue
        try:
            data = json.loads(result_file.read_text())
        except json.JSONDecodeError:
            print(f"[WARN] Skipping malformed {result_file}")
            continue

        # Defect-type extraction stats (may be absent in older evals)
        dt = data.get("defect_type_scoring", {})

        rows.append({
            "step":         _step_from_name(subdir.name),
            "dir":          subdir.name,
            "path":         subdir,
            "n_samples":    data.get("n_samples",  "?"),
            "accuracy":     data.get("accuracy",   None),
            "precision":    data.get("precision",  None),
            "recall":       data.get("recall",     None),
            "f1":           data.get("f1",         None),
            "roc_auc":      data.get("roc_auc",    None),
            "specificity":  data.get("specificity", None),
            # Defect extraction
            "dt_exact":     dt.get("exact_match_rate", None),
            "dt_fuzzy":     dt.get("avg_fuzzy_score",  None),
            "dt_extracted": dt.get("n_both_extracted", None),
            "dt_total":     dt.get("n_true_anomalies", None),
            # Confusion matrix raw counts
            "cm":           data.get("confusion_matrix", {}),
            # Per-category (kept for --per-category)
            "_data":        data,
        })
    return rows


def _fmt(val, width: int, decimals: int = 4) -> str:
    if val is None:
        return "—".center(width)
    if isinstance(val, float):
        return f"{val:.{decimals}f}".rjust(width)
    return str(val).rjust(width)


def _print_table(rows: list[dict], best_step: int, show_roc: bool = True) -> None:
    cols = ["step", "n_samples", "accuracy", "precision", "recall", "f1"]
    if show_roc and any(r["roc_auc"] is not None for r in rows):
        cols.append("roc_auc")

    header = "  ".join(c.rjust(COLUMN_WIDTHS.get(c, 8)) for c in cols)
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for row in rows:
        marker = "  <- best F1" if row["step"] == best_step else ""
        line = "  ".join(
            _fmt(row.get(c), COLUMN_WIDTHS.get(c, 8))
            for c in cols
        )
        print(f"{line}{marker}")
    print(sep)


def _print_defect_stats(rows: list[dict]) -> None:
    """Print defect-type extraction stats for each checkpoint."""
    any_data = any(r["dt_exact"] is not None for r in rows)
    if not any_data:
        print("  (No defect-type extraction data — re-run vlm-eval to populate.)")
        return

    header = f"  {'step':>6}  {'extracted':>9}  {'total':>7}  {'exact%':>7}  {'fuzzy':>6}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in rows:
        if r["dt_exact"] is None:
            continue
        pct = f"{r['dt_exact']:.1%}"
        fuz = f"{r['dt_fuzzy']:.3f}" if r["dt_fuzzy"] is not None else "—"
        ext = str(r["dt_extracted"]) if r["dt_extracted"] is not None else "—"
        tot = str(r["dt_total"]) if r["dt_total"] is not None else "—"
        print(f"  {r['step']:>6}  {ext:>9}  {tot:>7}  {pct:>7}  {fuz:>6}")


def _print_per_category(rows: list[dict]) -> None:
    """Print per-category F1/recall for each checkpoint that has the data."""
    any_printed = False
    for row in rows:
        per_cat = row["_data"].get("per_category")
        if not per_cat:
            continue
        if not any_printed:
            print("\n-- Per-Category Breakdown -------------------------------------------")
            any_printed = True
        cm = row["cm"]
        print(
            f"\n  step {row['step']} ({row['dir']})  "
            f"TP={cm.get('tp','?')}  FP={cm.get('fp','?')}  "
            f"TN={cm.get('tn','?')}  FN={cm.get('fn','?')}"
        )
        print(f"  {'category':<20}  {'n':>4}  {'acc':>6}  {'rec':>6}  {'f1':>6}  {'spec':>6}")
        print(f"  {'-'*20}  {'-'*4}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}")
        for cat, m in sorted(per_cat.items()):
            n    = m.get("n_samples", "?")
            f1   = _fmt(m.get("f1"),          6, 3)
            acc  = _fmt(m.get("accuracy"),     6, 3)
            rec  = _fmt(m.get("recall"),       6, 3)
            spec = _fmt(m.get("specificity"),  6, 3)
            print(f"  {cat:<20}  {str(n):>4}  {acc}  {rec}  {f1}  {spec}")


def _print_version_comparison(version_dirs: list[Path]) -> None:
    """Print a side-by-side summary of best checkpoints from each version."""
    print("\n" + "=" * 70)
    print("VERSION COMPARISON  (best checkpoint by F1 from each run)")
    print("=" * 70)
    print(f"  {'version':<35}  {'F1':>6}  {'Rec':>6}  {'Prec':>6}  {'AUC':>6}  {'step':>5}")
    print(f"  {'-'*35}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*5}")

    for vdir in version_dirs:
        if not vdir.is_dir():
            print(f"  {str(vdir):<35}  [directory not found]")
            continue
        rows = _load_results(vdir)
        if not rows:
            print(f"  {vdir.name:<35}  [no eval_results.json found]")
            continue
        scored = [r for r in rows if r["f1"] is not None]
        if not scored:
            print(f"  {vdir.name:<35}  [no F1 scores — run vlm-eval]")
            continue
        best = max(scored, key=lambda r: r["f1"])
        f1   = _fmt(best["f1"],         6, 4)
        rec  = _fmt(best["recall"],     6, 4)
        prec = _fmt(best["precision"],  6, 4)
        auc  = _fmt(best["roc_auc"],    6, 4)
        print(
            f"  {vdir.name:<35}  {f1}  {rec}  {prec}  {auc}  {best['step']:>5}"
        )

    print("=" * 70)
    print(
        "\nNote: Run  vlm-eval <checkpoint-dir> <config>  on each checkpoint\n"
        "to populate eval_results.json before using --compare-versions."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare eval_results.json across checkpoints and training versions"
    )
    parser.add_argument(
        "checkpoint_roots",
        type=Path,
        nargs="+",
        help=(
            "One or more checkpoint directories to scan. "
            "When multiple are given, --compare-versions is implied."
        ),
    )
    parser.add_argument(
        "--sort",
        choices=["step", *METRICS],
        default="step",
        help="Column to sort by (default: step)",
    )
    parser.add_argument(
        "--desc",
        action="store_true",
        help="Sort descending (default: ascending for step, descending for metrics)",
    )
    parser.add_argument(
        "--per-category",
        action="store_true",
        help="Also print per-category F1/recall breakdown for each checkpoint",
    )
    parser.add_argument(
        "--defect-stats",
        action="store_true",
        help="Print defect-type extraction exact-match and fuzzy-score stats",
    )
    parser.add_argument(
        "--compare-versions",
        action="store_true",
        help=(
            "Print a cross-version summary table (best checkpoint per directory). "
            "Automatically enabled when multiple directories are provided."
        ),
    )
    args = parser.parse_args()

    multi = len(args.checkpoint_roots) > 1

    # ── Per-directory tables ─────────────────────────────────────────────────
    for checkpoint_root in args.checkpoint_roots:
        if not checkpoint_root.is_dir():
            print(f"[ERROR] Directory not found: {checkpoint_root}")
            continue

        rows = _load_results(checkpoint_root)

        if not rows:
            print(f"\nNo eval_results.json found under {checkpoint_root}")
            print("Run: vlm-eval <checkpoint-dir> <config.yaml>")
            continue

        # Sort
        reverse = args.desc if args.sort == "step" else not args.desc
        rows.sort(
            key=lambda r: (r[args.sort] is None, r[args.sort] or 0),
            reverse=reverse,
        )

        # Best F1
        scored   = [r for r in rows if r["f1"] is not None]
        best_row = max(scored, key=lambda r: r["f1"]) if scored else None
        best_step = best_row["step"] if best_row else -1

        print(f"\nCheckpoint root: {checkpoint_root.resolve()}")
        print(f"Checkpoints with results: {len(rows)}\n")
        _print_table(rows, best_step)

        if best_row:
            cm = best_row["cm"]
            print(
                f"\nBest F1={best_row['f1']:.4f} at step {best_step}  ({best_row['dir']})"
            )
            if cm:
                print(
                    f"  Confusion: TP={cm.get('tp','?')}  FP={cm.get('fp','?')}  "
                    f"TN={cm.get('tn','?')}  FN={cm.get('fn','?')}"
                )
            print(
                f"  To push: python scripts/push_to_hub.py "
                f"--checkpoint {checkpoint_root / best_row['dir']} "
                f"--repo-id your-username/llava-mvtec-defect-detection "
                f"--config configs/local_8gb.yaml"
            )

        if args.defect_stats:
            print("\n-- Defect-Type Extraction -------------------------------------------")
            _print_defect_stats(rows)

        if args.per_category:
            _print_per_category(rows)

    # ── Cross-version comparison ─────────────────────────────────────────────
    if multi or args.compare_versions:
        _print_version_comparison(args.checkpoint_roots)


if __name__ == "__main__":
    main()
