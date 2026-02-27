#!/usr/bin/env python3
"""Quick environment check — run after `pip install -e .` or `make install`."""

import sys

print(f"Python {sys.version}")

checks = [
    ("torch",         lambda m: f"{m.__version__}  CUDA={m.cuda.is_available()}"),
    ("torchvision",   lambda m: m.__version__),
    ("transformers",  lambda m: m.__version__),
    ("peft",          lambda m: m.__version__),
    ("accelerate",    lambda m: m.__version__),
    ("bitsandbytes",  lambda m: m.__version__),
    ("wandb",         lambda m: m.__version__),
    ("yaml",          lambda m: m.__version__),
]

ok = True
for name, info_fn in checks:
    try:
        import importlib
        mod = importlib.import_module(name)
        print(f"  OK  {name:<16} {info_fn(mod)}")
    except Exception as e:
        print(f"  FAIL {name:<15} {e}")
        ok = False

sys.exit(0 if ok else 1)
