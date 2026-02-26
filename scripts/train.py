"""
Training entry point.

Usage:
    # Local 8 GB GPU
    python scripts/train.py --config configs/local_8gb.yaml

    # Colab A100 (run inside notebook)
    python scripts/train.py --config configs/colab_a100.yaml

    # Override any config value from CLI
    python scripts/train.py --config configs/local_8gb.yaml training.epochs=5 lora.r=16
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the repo root importable regardless of how the script is invoked
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.vlm_defect.config import load_config
from src.vlm_defect.trainer import run_training


def apply_overrides(cfg, overrides: list[str]) -> None:
    """Allow `key.subkey=value` overrides from CLI (basic dotted-path setter)."""
    for override in overrides:
        if "=" not in override:
            print(f"[warn] ignoring malformed override: {override!r}")
            continue
        path, value = override.split("=", 1)
        keys = path.split(".")
        obj = cfg
        for k in keys[:-1]:
            obj = getattr(obj, k)
        leaf = keys[-1]
        current = getattr(obj, leaf)
        # Cast to the same type as the existing field
        try:
            if isinstance(current, bool):
                typed = value.lower() in ("1", "true", "yes")
            else:
                typed = type(current)(value)
        except (ValueError, TypeError):
            typed = value
        setattr(obj, leaf, typed)
        print(f"  override: {path} = {typed!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune LLaVA on MVTec-AD")
    parser.add_argument(
        "--config",
        default="configs/local_8gb.yaml",
        help="Path to hardware profile YAML",
    )
    args, overrides = parser.parse_known_args()

    cfg = load_config(args.config)
    if overrides:
        print("Applying CLI overrides:")
        apply_overrides(cfg, overrides)

    run_training(cfg)


if __name__ == "__main__":
    main()
