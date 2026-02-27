"""CLI entrypoints declared in pyproject.toml.

    vlm-prepare [--dataset-root PATH] [--output PATH]
    vlm-train   <config.yaml> [KEY=VALUE ...]
    vlm-eval    <checkpoint-dir> <config.yaml>
"""

from vlm_defect.data import main as _prepare_main
from vlm_defect.trainer import main as _train_main


def prepare() -> None:
    """Entrypoint for ``vlm-prepare``."""
    _prepare_main()


def train() -> None:
    """Entrypoint for ``vlm-train``."""
    _train_main()


def eval() -> None:
    """Entrypoint for ``vlm-eval``."""
    from vlm_defect.evaluate import main as _eval_main
    _eval_main()
