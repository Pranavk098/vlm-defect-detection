"""
Pre-training file verification for VLM defect detection.

Checks that all required files and directories are present before
the training process starts, providing clear error messages for
anything that is missing.
"""

import sys
from pathlib import Path


REQUIRED_FILES = [
    "requirements.txt",
    "mvtec_train.json",
    "src/vlm_defect/train.py",
    "src/vlm_defect/prepare_dataset.py",
    "src/vlm_defect/verify_install.py",
]

REQUIRED_DIRS = [
    "LLaVA",
    "mvtec_anomaly_detection",
]

REQUIRED_LLAVA_FILES = [
    "LLaVA/llava/train/train_mem.py",
]


def verify_all_files_present(project_dir: Path | None = None, exit_on_failure: bool = False) -> bool:
    """
    Verify all required files and directories exist before training.

    Args:
        project_dir: Root of the project. Defaults to the repo root
                     (two levels above this file).
        exit_on_failure: If True, call sys.exit(1) when verification fails.

    Returns:
        True if every required path is present, False otherwise.
    """
    if project_dir is None:
        # src/vlm_defect/verify_files.py  ->  repo root is two parents up
        project_dir = Path(__file__).resolve().parent.parent.parent

    print(f"[verify] Project root: {project_dir}")
    print("[verify] Checking required files and directories ...\n")

    missing: list[str] = []

    # --- required files ---
    for rel_path in REQUIRED_FILES:
        full_path = project_dir / rel_path
        if full_path.exists():
            print(f"  [OK]  {rel_path}")
        else:
            print(f"  [MISSING]  {rel_path}")
            missing.append(rel_path)

    print()

    # --- required directories ---
    for rel_dir in REQUIRED_DIRS:
        full_path = project_dir / rel_dir
        if full_path.is_dir():
            print(f"  [OK]  {rel_dir}/")
        else:
            print(f"  [MISSING]  {rel_dir}/  (directory)")
            missing.append(f"{rel_dir}/")

    print()

    # --- LLaVA internal files (only checked when LLaVA dir exists) ---
    for rel_path in REQUIRED_LLAVA_FILES:
        full_path = project_dir / rel_path
        if full_path.exists():
            print(f"  [OK]  {rel_path}")
        else:
            print(f"  [MISSING]  {rel_path}")
            missing.append(rel_path)

    print()

    if missing:
        print(f"[verify] FAILED — {len(missing)} item(s) missing:")
        for item in missing:
            print(f"         - {item}")
        print()
        print("[verify] Resolve the missing items before starting training.")
        if exit_on_failure:
            sys.exit(1)
        return False

    print("[verify] All required files are present. Ready to train.")
    return True


if __name__ == "__main__":
    verify_all_files_present(exit_on_failure=True)
