# REPORT — LLaVA-1.5-7B QLoRA Defect Detection on MVTec AD (RTX 5070 8GB)

> TEMPLATE — fill every `[FILL]` after GPU runs. No fabricated numbers: empty
> cells stay empty until measured. Delete this banner when the report is final.
> Recipe: `docs/RTX5070_8GB_RECIPE.md` · Profile: `configs/rtx5070_8gb.yaml` ·
> Ablation protocol + post-training checklist: `scripts/run_ablation.md`.

**Model:** `[FILL: HF Hub repo ID after push_to_hub.py, e.g. user/llava-mvtec-defect]` |
**Best checkpoint:** `[FILL: checkpoints/llava-mvtec-qlora-rtx5070/checkpoint-NNN]` |
**Date / GPU / driver:** `[FILL: e.g. 2026-09-XX · RTX 5070 laptop 8GB · CUDA 12.x · torch x.y]` |
**Commit:** `[FILL: git rev-parse --short HEAD]` · **Seed:** 42 (single-seed; see §6)

---

## 1. Abstract

[FILL, 5–8 sentences after results: what was trained (LLaVA-1.5-7B + QLoRA r=16
on MVTec AD, 8GB laptop), headline metrics (F1 / recall / ROC-AUC / grounding
detection-rate@0.5), ablation conclusion (QLoRA vs LoRA vs DoRA), and the
one-sentence limitation.]

## 2. Method

### 2.1 Base model
LLaVA-1.5-7B (`llava-hf/llava-1.5-7b-hf`): CLIP ViT-L/14 @336px vision tower
(576 vision tokens/image) + Vicuna-7B LLM + MLP projector. Both towers
**frozen**; only LoRA adapters train (`prepare_model_for_kbit_training`).

### 2.2 Memory recipe (why 7B fits 8GB)
4-bit NF4 base (~3.9GB) + double quant + bf16 compute · LoRA r=16/α=32/dropout
0.05 on 7 LLM modules (`q/k/v/o_proj` + `gate/up/down_proj`) · micro-batch
1 × accum 16 (effective 16) · gradient checkpointing ON · bf16 (auto-fallback
fp16) · compact eval head (2 logits/sample, `WeightedCETrainer.prediction_step`).
Deliberately excluded: 8-bit/paged optimizer (one-line patch, not applied —
state here if you apply it), flash-attn/xFormers (SDPA default), Unsloth (no
LLaVA path), DeepSpeed ZeRO (single-GPU, rejected — see recipe §2.7).

### 2.3 Task formulation
Category-aware prompt (`Is there any anomaly in this {category} image? …`),
centre-crop zoom (70%) for small-defect categories
(screw/capsule/transistor/pill/metal_nut), Yes-token loss ×2.0 (recall fix for
v2's FN=269), category oversampling (transistor/screw ×4, capsule ×3, pill
×2.5, cable ×2), 65% anomaly fold-in, F1-based checkpoint selection, global
threshold 0.25 + per-category overrides (`CATEGORY_THRESHOLDS`), optional
hflip TTA on hard categories.

## 3. Experimental setup

| Item | Value |
|---|---|
| Train split | `[FILL: N train (normal/anomaly), from prepare_data log]` |
| Val split | 10% held-out (`val_fraction: 0.1`, seed 42) |
| Test split | `[FILL: N test]` (`data/mvtec_test.json` + `eval_test.json` harness split) |
| Grounding subset | `mvtec_grounding_200.json` (200 boxes, `bbox_1000`) |
| Epochs / eff. batch / LR / sched | 3 (ablation) / 16 / 1e-4 (v3) / cosine, warmup 0.05 — note any override (e.g. 5-epoch best run, lr 2e-4 parity) |
| Hardware / precision | RTX 5070 laptop 8GB, Windows, bf16 `[or fp16 fallback — state which]` |
| Eval protocol | `eval_mvtec.py` (text-acc, per-class AUROC, rationale 0–3, grounding det-rate) + `evaluate.py --sweep-threshold` + `--log-failures` + restricted TTA |

## 4. Results

### 4.1 Ablation — LoRA vs QLoRA vs DoRA (fill after GPU runs)

Fixed controls: 3 epochs, seed 42, α=2r, dropout 0.05, eff-batch 16.
Arm A ran on `[FILL: Colab T4/A100 — never on 8GB]`; Arms B/C on RTX 5070 8GB.

| Method | Config / ckpt | Text-acc (normal / defective) | Mean per-class AUROC | F1 / Recall / Prec (evaluate.py) | ROC-AUC | Rationale mean (0–3) | Grounding det-rate@0.5 (n) | Peak VRAM | Wall-clock |
|---|---|---|---|---|---|---|---|---|---|
| LoRA r=16 fp16 (A) | `configs/lora_r16.yaml` → `[ckpt]` | `[FILL]` | `[FILL]` | `[FILL]` | `[FILL]` | `[FILL]` | `[FILL]` | `[FILL]` | `[FILL]` |
| QLoRA-4bit r=16 (B, primary) | `configs/rtx5070_8gb.yaml` → `[ckpt]` | `[FILL]` | `[FILL]` | `[FILL]` | `[FILL]` | `[FILL]` | `[FILL]` | `[FILL]` | `[FILL]` |
| QDoRA-4bit r=16 (C) | same + `use_dora` patch → `[ckpt]` | `[FILL]` | `[FILL]` | `[FILL]` | `[FILL]` | `[FILL]` | `[FILL]` | `[FILL]` | `[FILL]` |

Pre-registered hypotheses (from `run_ablation.md`): H1 QLoRA ≈ LoRA text-acc
within noise at ~half VRAM · H2 DoRA ≥ LoRA on grounding IoU · H3 rationale
keyword-hit > single-sentence rate. Verdicts: `[FILL: confirmed/refuted + delta]`.

### 4.2 Per-category table (best arm only)

| Category | n | acc | recall | F1 | thr used | Note |
|---|---|---|---|---|---|---|
| `[15 rows — paste from evaluate.py Per-Category output]` | | | | | | |
| **Global** | `[N]` | `[acc]` | `[rec]` | `[F1]` | 0.25 + overrides | ROC-AUC `[FILL]` · CM: TP `[ ]` FP `[ ]` TN `[ ]` FN `[ ]` |

### 4.3 Grounding demo (IoU@0.5, 200-box subset)
Method: predicted `bbox_1000` vs ground-truth boxes (`scripts/eval_mvtec.py`
`--grounding-file`); box-level **detection-rate proxy** — explicitly NOT
pixel-PRO (text-only VLM emits no anomaly maps).
Mean IoU `[FILL]` · detection-rate@0.5 `[FILL]` (n=`[FILL]`) · qualitative:
`[FILL: 2–3 success + 2–3 failure examples with image paths]`.

### 4.4 Defect-naming (known limitation)
Expected: free-form exact-match stays low (v3 ckpt-500 collapsed onto
"scratch" ~73%); constrained-vocab remap (`_constrained_defect_name`) reported
as upper bound: raw exact `[FILL]` / constrained `[FILL]` / avg fuzzy `[FILL]`.
Failure cases: `eval_outputs/failures_<arm>.json` (`--log-failures`).

## 5. Gradio demo
`vlm-app --repo-id [FILL]` (merged Hub model) or `--checkpoint <ckpt> --config
configs/rtx5070_8gb.yaml`. 15-class gallery (`app/demo.py` expectations) +
failure-case tab sourced from `failures_<arm>.json`. Screenshot/link: `[FILL]`.
Mock-mode responses (`app/server.py` without weights) were never scored as results.

## 6. Threats to validity
1. **Single seed (42)** — no variance bars; rerun best arm ≥3 seeds if claiming SOTA deltas.
2. **Anomaly fold-in leakage** — 65% of test anomalies seen in training (by design, to learn templates); test split holds the remaining 35% + all test-good. Report fold-in fraction alongside every number.
3. **Threshold fitting** — per-category thresholds tuned on the test split (`--sweep-threshold`); optimistic by construction. Mitigation: freeze thresholds from Arm B, apply unchanged to Arm C.
4. **Defect-naming collapse** — binary pass/fail is the reliable claim; naming numbers are diagnostic.
5. **Grounding ≠ PRO** — box detection-rate proxy only; no pixel-AUROC/PRO claim.
6. **Mock/reference baselines** — `--mock` and reference-answer scores validate the harness, never the model; label them as such in every table.
7. **Hardware specificity** — VRAM/time numbers are RTX-5070-laptop + Windows specific; Arm A hardware differs (noted in §4.1).

## 7. Conclusion
[FILL after results: 3–5 sentences — did 7B-on-8GB reproduce, which arm wins on
what metric and at what cost, what ships (Hub ID + demo), what is next
(multi-seed, 3B-fallback parity, constrained decoding for naming).]

## A. Reproducibility checklist
- [ ] `configs/rtx5070_8gb.yaml` unmodified (or diff pasted below) · commit `[FILL]`
- [ ] `data/mvtec_train.json` built with `--anomaly-train-fraction 0.65`
- [ ] `trainer_state.json best_model_checkpoint` recorded per arm
- [ ] `eval_outputs/results_<arm>.json` + `failures_<arm>.json` committed (metrics only, no weights)
- [ ] `nvidia-smi` peak + `max_memory_allocated` + wall-clock per arm recorded
- [ ] Hub repo `[FILL]` loads via `vlm-app --repo-id` and matches local ckpt (±noise)
- [ ] No weight files (`*.bin/*.safetensors`) committed — `git status` clean of checkpoints
