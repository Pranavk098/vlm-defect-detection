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

## Post-training checklist (RTX 5070 8GB 2nd cycle — do in order, check off)

> Primary profile is now `configs/rtx5070_8gb.yaml` (Arm B, Windows-safe).
> Legacy `configs/{lora_r16,qlora_4bit,dora_r16}.yaml` use the OLD schema
> (`train_file`/`eval_file`, `liuhaotian/llava-v1.5-7b` base) and do NOT plug
> into `src/vlm_defect/trainer.py` directly — use the override commands in
> `docs/RTX5070_8GB_RECIPE.md` §3 instead. Threshold/TTA flags below refer to
> `src/vlm_defect/evaluate.py`.

- [ ] **(a) Merge LoRA + push to Hub** — best ckpt per arm (see
  `trainer_state.json` → `best_model_checkpoint`):
  ```powershell
  python scripts/push_to_hub.py --checkpoint checkpoints/llava-mvtec-qlora-rtx5070/checkpoint-NNN `
    --repo-id <user>/llava-mvtec-defect-detection --config configs/rtx5070_8gb.yaml --device cpu
  ```
  `--device cpu` is REQUIRED on 8GB (bf16 merge needs ~14GB; CPU RAM covers
  it). Then verify parity: `vlm-app --repo-id <user>/llava-mvtec-defect-detection`
  vs local ckpt on 5 probe images. Never commit `*.safetensors`/`*.bin`.
- [ ] **(b) Gradio demo — 15-class gallery + failure cases:**
  `vlm-app --repo-id <repo>` (or `--checkpoint <ckpt> --config
  configs/rtx5070_8gb.yaml`); gallery covers all 15 MVTec categories
  (`app/demo.py` category list); add a failure-case tab sourced from
  `eval_outputs/failures_<arm>.json` (`evaluate.py --log-failures`).
  Screenshot + repo link go into `REPORT_template.md` §5.
- [ ] **(c) Ablation table** — one row per arm (LoRA r16 / QLoRA r16 / QDoRA r16):
  text-acc (normal/defective), mean per-class AUROC, F1/recall/precision,
  ROC-AUC, rationale mean (0–3), grounding det-rate@0.5, peak VRAM,
  wall-clock. Template lives in `REPORT_template.md` §4.1; hypotheses H1–H3
  verdicts recorded with deltas, not vibes.
- [ ] **(d) Grounding demo IoU@0.5** — `eval_mvtec.py --preds
  eval_outputs/preds_<arm>.json --grounding-file mvtec_grounding_200.json`;
  report mean IoU + detection-rate@IoU≥0.5 as a **box-level PRO proxy** (never
  as pixel-PRO — text-only VLM emits no anomaly maps).
- [ ] **(e) REPORT.md** — copy `REPORT_template.md` → `REPORT.md`, fill every
  `[FILL]`, record commit hash + `nvidia-smi` peak + wall-clock per arm,
  freeze Arm-B thresholds before scoring Arm C (anti-leakage), list threats
  (§6) honestly. Metrics JSONs (`eval_outputs/results_<arm>.json`) committed;
  weights never committed.
