#!/usr/bin/env python3
"""Thin wrapper — delegates to vlm_defect.data.

Usage:
    python scripts/prepare_data.py                            # defaults
    python scripts/prepare_data.py --dataset-root /data/mvtec --output data/mvtec_train.json
    make prepare                                              # via Makefile
    vlm-prepare                                              # installed entrypoint
"""

from vlm_defect.data import main

if __name__ == "__main__":
    main()
