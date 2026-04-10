#!/usr/bin/env python3
"""Thin wrapper — delegates to vlm_defect.trainer.

Usage:
    python scripts/train.py configs/local_8gb.yaml
    python scripts/train.py configs/local_8gb.yaml training.report_to=wandb
    make train                                               # via Makefile
    make train OVERRIDE="training.report_to=wandb"
    vlm-train configs/local_8gb.yaml                        # installed entrypoint
"""

from vlm_defect.trainer import main

if __name__ == "__main__":
    main()
