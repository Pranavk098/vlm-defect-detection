#!/usr/bin/env python3
"""Thin wrapper — delegates to vlm_defect.evaluate.

Usage:
    python scripts/evaluate.py checkpoints/llava-mvtec-lora configs/local_8gb.yaml
    make eval CHECKPOINT=checkpoints/llava-mvtec-lora
    vlm-eval checkpoints/llava-mvtec-lora configs/local_8gb.yaml
"""

from vlm_defect.evaluate import main

if __name__ == "__main__":
    main()
