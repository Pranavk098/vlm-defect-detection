"""CLI entrypoints declared in pyproject.toml.

    vlm-prepare [--dataset-root PATH] [--output-train PATH] [--output-test PATH]
    vlm-train   <config.yaml> [KEY=VALUE ...]
    vlm-eval    <checkpoint-dir> <config.yaml>
    vlm-app     --repo-id REPO_ID | --checkpoint DIR [--config YAML] [--share] [--port N]

All heavy imports (transformers, peft, torch training stack) are deferred to
inside each function so that ``import vlm_defect.cli`` is cheap and works
without the full [train] extras installed (e.g. in CI import checks).
"""


def prepare() -> None:
    """Entrypoint for ``vlm-prepare``."""
    from vlm_defect.data import main as _prepare_main

    _prepare_main()


def train() -> None:
    """Entrypoint for ``vlm-train``."""
    from vlm_defect.trainer import main as _train_main

    _train_main()


def eval() -> None:
    """Entrypoint for ``vlm-eval``."""
    from vlm_defect.evaluate import main as _eval_main

    _eval_main()


def app() -> None:
    """Entrypoint for ``vlm-app``."""
    from vlm_defect.app import main as _app_main

    _app_main()
