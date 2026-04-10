"""CLI entrypoints declared in pyproject.toml.

    vlm-prepare [--dataset-root PATH] [--output-train PATH] [--output-test PATH]
    vlm-train   <config.yaml> [KEY=VALUE ...]
    vlm-eval    <checkpoint-dir> <config.yaml>
    vlm-app     --repo-id REPO_ID | --checkpoint DIR [--config YAML] [--share] [--port N]
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


def app() -> None:
    """Entrypoint for ``vlm-app``."""
    from vlm_defect.app import main as _app_main
    _app_main()
