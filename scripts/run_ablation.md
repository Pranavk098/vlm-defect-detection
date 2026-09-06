# Efficient-Tuning Ablation — 1-GPU Protocol

Compares **LoRA (r=16)** vs **QLoRA 4-bit (r=64)** vs **DoRA (r=16)** on
MVTec defect classification + grounding. Everything except the method is
fixed so the comparison is fair.

## Fixed controls

| Hyperparameter | Value |
|---|---|
| Base model | `liuhaotian/llava-v1.5-7b` |
| Epochs | 3 |
| Learning rate | 2e-4, cosine, warmup 0.03 |
| Batch | 4 per-device × 4 grad-accum (effective 16) |
| Alpha rule | alpha = 2×r in every arm |
| Dropout | 0.05 |
| Seed | 42 (`--seed 42`, `PYTHONHASHSEED=42`) |
| Eval | every 100 steps, best ckpt on `eval_loss` |

## Steps (Colab T4 16GB, or A100 for the LoRA arm)

```bash
# 0. Data + held-out split (run once)
python prepare_mvtec_json.py
python scripts/make_eval_split.py --input mvtec_train.json --output eval_test.json
python scripts/make_grounding_subset.py --root mvtec_anomaly_detection --output mvtec_grounding_200.json

# 1a. Arm A — LoRA r=16 (needs ~2x VRAM; A100 preferred, or halve batch + double accum on T4)
python train_mvtec.py --config configs/lora_r16.yaml

# 1b. Arm B — QLoRA 4-bit r=64 (fits T4-16GB; this is the default path)
python train_mvtec.py --config configs/qlora_4bit.yaml

# 1c. Arm C — DoRA r=16 (requires peft>=0.9 AND a one-line patch:
#     add use_dora=True to the LoraConfig in LLaVA/llava/train/train.py —
#     upstream has no flag for it; train_mvtec.py warns if unpatched)
python train_mvtec.py --config configs/dora_r16.yaml

# 2. Eval each arm (collect preds.json per arm first — see app/server.py POST /infer —
#    or score the reference answers for a harness smoke test with --mock)
python scripts/eval_mvtec.py --eval-file eval_test.json --preds eval_outputs/preds_<arm>.json \
    --grounding-file mvtec_grounding_200.json --output eval_outputs/results_<arm>.json
```

## Upstream patches required (honest list)

1. **Eval wiring**: stock `LLaVA/llava/train/train.py` hardcodes
   `eval_dataset=None`, so `evaluation_strategy=steps` needs a ~3-line patch
   in `make_supervised_data_module` (load `eval_test.json` into a second
   `LazySupervisedDataset`, accept `--eval_data_path`). Without it, training
   runs but never evaluates mid-run — use `scripts/eval_mvtec.py` standalone.
2. **DoRA flag**: same file builds a plain `LoraConfig`; add `use_dora=True`.

## Metric table template (fill after GPU runs — no numbers yet)

| Method | Config | Text-acc (normal / defective) | Mean per-class AUROC | IoU@0.5 (grounding-200) | Rationale (0–3) | Peak VRAM | Time |
|---|---|---|---|---|---|---|---|
| LoRA r=16 | `configs/lora_r16.yaml` | / | | | | | |
| QLoRA 4-bit r=64 | `configs/qlora_4bit.yaml` | / | | | | | |
| DoRA r=16 | `configs/dora_r16.yaml` | / | | | | | |

## Hypotheses (pre-registered)

- H1: QLoRA-4bit matches LoRA text-accuracy within noise at ~half the VRAM.
- H2: DoRA ≥ LoRA on grounding IoU (direction+magnitude helps localization).
- H3: All arms score rationale keyword-hit > single-sentence rate (models
  name defects but ramble) — motivating the one-sentence prompt constraint.
