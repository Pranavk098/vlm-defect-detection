# RTX 5070-Laptop 8GB Recipe — LLaVA-1.5-7B QLoRA on MVTec AD

Target: laptop RTX 5070 (~8GB VRAM), ~8GB RAM budget, Windows, **no GPU runs in
this commit** — this file is the executable recipe for the user to run.
Primary config: `configs/rtx5070_8gb.yaml` (Arm B — the only arm that fits 8GB).

Contents: §1 v3 8GB inventory · §2 method ranking · §3 recipe commands ·
§4 VRAM curves · §5 checkpoint/resume on Windows · §6 eval protocol ·
§7 time estimates · §8 fallbacks.

---

## §1 — v3 8GB inventory (what the rebased codebase already does)

Derived from `configs/local_8gb.yaml`, `src/vlm_defect/model.py`,
`src/vlm_defect/trainer.py`, `src/vlm_defect/data.py`, `scripts/train.py`,
`scripts/evaluate.py`, `SETUP.md`. Verified by reading, not by running.

| Technique | v3 status | Where |
|---|---|---|
| 4-bit QLoRA, NF4 + double quant, bf16 compute | ✅ YES | `model.py:26-31`, config `quantization:` |
| `prepare_model_for_kbit_training` (freezes base, enables grad-ckpt path) | ✅ YES | `model.py:39` |
| LoRA r=16, α=32, dropout 0.05 | ✅ YES | config `lora:` |
| LoRA on attention **+ MLP** (`q/k/v/o` + `gate/up/down_proj`) | ✅ YES (v3 change) | config `lora.target_modules` |
| Gradient checkpointing | ✅ YES | config `gradient_checkpointing: true` |
| Micro-batch 1 × accum 16 (effective 16) | ✅ YES | config `training:` |
| bf16 with **auto-fallback to fp16** if GPU lacks bf16 | ✅ YES | `trainer.py:491-492` (RTX 5070 has bf16, so bf16 is used) |
| LLM + vision tower frozen, adapters-only training | ✅ YES (base frozen; LoRA targets LLM attn/MLP only — projector and CLIP tower train nothing) | `model.py:39-52` |
| F1-based checkpoint selection (`metric_for_best_model: f1`) | ✅ YES (v3 change) | config + `trainer.py:102-166` |
| Memory-efficient eval (`prediction_step` returns 2 logits/sample, not B×T×V) | ✅ YES (v3 change) | `trainer.py:273-353` |
| Yes-token upweight ×2.0 (recall fix), category oversampling, 65% anomaly fraction | ✅ YES (v3 changes) | config `class_balance:` / `category_weights:` / `data:` |
| **Paged / 8-bit AdamW** | ❌ NO — HF default AdamW (fp32 m+v on ~40M LoRA params ≈ 0.3GB). 8-bit would save ~0.2GB + absorb spikes; needs one-line `TrainingArguments(optim=...)` patch | `trainer.py:494-520` has no `optim` arg |
| **flash-attn / xFormers pinning** | ❌ NO — `model.py` passes no `attn_implementation`; HF default (SDPA on torch≥2.0) applies | `model.py:33-38` |
| **Unsloth fast kernels** | ❌ NO — no Unsloth path in codebase (and no LLaVA-1.5 fast-model support to drop in) | — |
| **DeepSpeed ZeRO offload** | ❌ NO (correctly — see §2.7) | — |
| Windows-safe dataloader (`num_workers=0`) | ❌ NO — v3 sets 4 (hang risk on Windows); **fixed to 0 in `rtx5070_8gb.yaml`** | config `training:` |
| `report_to` default | ⚠️ v3 defaults to `"wandb"` (errors without login); **rtx5070 profile defaults to `"none"`** | config `training:` |

`SETUP.md` is stale in two places: the sample YAML block still shows
`lora.r: 128 / batch 4` (actual v3 file uses r=16 / batch 1), and it claims
"bitsandbytes has no native Windows build — use WSL2/Docker". bitsandbytes now
ships official Windows x86-64 wheels (CUDA 11.8–12.6+, VS2022, Python ≥3.10 —
see §2.7 sources), so native Windows training is viable; WSL2/Docker remains
the lower-friction fallback if the Windows bnb/CUDA combo fails.

---

## §2 — 8GB method ranking for LLaVA-1.5-7B (2026 research)

Applies-to-LLaVA note up front: the vision tower (CLIP ViT-L/14 @336px → 576
vision tokens/sample) makes VLM activation memory strictly larger than the
LLM-only numbers most guides quote. Everything below assumes micro-batch 1,
seq ≤2048 (real MVTec seq ≈ 730 tokens), grad-ckpt ON.

| Rank | Method | Why here | LLaVA-1.5-7B verdict |
|---|---|---|---|
| 1 | **QLoRA 4-bit NF4 + double quant, bf16 compute** | Base 7B: ~14GB → ~3.9GB; adapters-only optimizer state (~0.3GB vs ~56GB full-FT). Within 1–2% of fp16-LoRA on benchmarks. | **Use it (Arm B).** Already wired in `model.py`. [Dettmers et al. QLoRA paper; MangoDev field guide 2026; OneRuby QLoRA guide 2026; ToolsKu 2026 pipeline] |
| 2 | **Gradient checkpointing** | ~30–50% activation saving for ~20% time cost. Non-negotiable at bs=1 on 8GB. | **ON** (both configs). [torchtune QLoRA docs; NeuralBase memory guide] |
| 3 | **Micro-batch 1–2 × accum 8–16 (effective 8–16)** | Same gradient math as big batches; only knob that moves activation memory linearly. | **1×16** (v3 + rtx5070). Never raise bs before trying accum. [MangoDev; InsiderLLM consumer recipe] |
| 4 | **LoRA r=16–32 on attention + MLP** | r=16 ≈ 40M params for 7B; MLP layers (`gate/up/down`) add ~30% params for measurable quality (QLoRA paper: all-linear targeting matters). r=64 doubles adapter VRAM for diminishing returns. | **r=16, 7 modules** (v3). r=32 only if F1 plateaus AND headroom allows. [OneRuby; ToolsKu r-table; InsiderLLM rank table] |
| 5 | **Paged / 8-bit AdamW (`paged_adamw_8bit`)** | Pages optimizer spikes to CPU via unified memory; ~25% throughput win measured on 8GB RTX 4060 (628 vs 500 tok/s) and enables 2048-token seqs within budget. | **Recommended one-line patch** to `trainer.py` (`optim=t.get("optim","adamw_torch")`); profiled as pure win on 8GB cards. [arXiv:2509.12229 RTX-4060 8GB study; ToolsKu paged-optimizer section] |
| 6 | **bf16 (not fp16) on Ada/Blackwell** | bf16 avoids fp16 overflow NaNs; RTX 5070 supports it. (One 8GB-class study found fp16 faster on RTX 4060 specifically — if loss NaNs or throughput disappoints, A/B `bf16:false`.) | **bf16:true** + existing auto-fallback. [MangoDev NaN guide; 2509.12229 fp16-vs-bf16 result] |
| 7 | **SDPA (default) > flash-attn/xFormers on this laptop** | Flash-attn2 has no Blackwell/Win wheels in most pins and buys ~1GB only on Ampere/Hopper; xFormers+flash-attn4 combos have silent-fallback hazards. SDPA (torch≥2.0 memory-efficient path) is zero-install and correct. | **Skip.** Revisit only with spare VRAM and a tested wheel. [Unsloth flash-attn4/xFormers issue #8957; CraftRigs Unsloth+LoRA guide] |
| 8 | **Unsloth fast kernels** | 2× faster / 30–50% less VRAM for supported LLM-only paths — but no LLaVA-1.5-7B fast-model path exists in this codebase, and porting the trainer to Unsloth is a rewrite, not a flag. | **Skip** for 2nd-cycle; note as future work in REPORT. [Unsloth repo/docs; Genαi July-2026 benchmark] |
| 9 | **DoRA / QDoRA (r=16)** | +0.6 on LLaVA-7B visual-instruction tuning (DoRA paper); +5–10% VRAM, +~44% wall-clock (norm ops unoptimized in PEFT ≤0.10). | **Arm C, second priority** after Arm B reproduces. Needs `use_dora=True` patch (see §3). [DoRA arXiv:2402.09353; NVIDIA DoRA blog; TildAlice LoRA-vs-DoRA bench; Spheron PEFT-2026 table] |
| 10 | **TinyLLaVA-3.1B / MobileVLM-3B fallback** | 3.1B model matches LLaVA-1.5-7B on generic VLM benches at ~half VRAM — but needs its own training path (SigLIP tower, different template) and MVTec-defect parity is unproven. | **Fallback only if 7B OOMs after §8 triage.** [TinyLLaVA arXiv:2402.14289; MobileVLM repo] |
| — | **DeepSpeed ZeRO offload** | Single-GPU laptop: no sharding peer, PCIe round-trip per step, known WSL2/quantized-weight interop bugs. Paged-optimizer + accum covers the same spike problem cheaper. | **Do NOT use.** Document as considered-and-rejected. [DeepSpeed ZeRO-Offload docs; bitsandbytes #1249 (ZeRO-3 × quant weights); DeepSpeed #5585 (WSL2 ZeRO-3)] |

Sources (all live, fetched Sep 2026):
1. Dettmers et al., *QLoRA: Efficient Finetuning of Quantized LLMs* (arXiv:2305.14314) — NF4 + double-quant + paged optimizers; 65B on 48GB, 7B within 1–2% of fp16.
2. Mango Developer, *Fine-Tuning Open-Source LLMs with LoRA and QLoRA: A Developer's Field Guide* (Aug 2026) — 7B QLoRA ≈ 6–8GB; r=16 + all-linear targets; bs=1×accum-16 on 8GB.
3. OneRuby/Kholodniak, *Fine-Tuning LLMs with QLoRA: Run a 7B Model on a Single GPU* (Jan 2026) — `paged_adamw_8bit` + grad-ckpt recipe; 7B QLoRA 6–10GB table.
4. arXiv:2509.12229 (Sep 2025), *Profiling LoRA/QLoRA on … 8GB VRAM* — RTX 4060 8GB: paged-8bit +25% throughput, 2048-token seqs fit at ~8.06GB, fp16-vs-bf16 A/B.
5. Spheron, *Beyond LoRA: DoRA, GaLore, PiSSA, VeRA* (May 2026) — QLoRA-r16-7B ≈ 6GB vs QDoRA ≈ 7GB; DoRA +5–10% VRAM table.
6. DoRA paper (arXiv:2402.09353, ICML-2024 oral) + NVIDIA DoRA blog — +0.6 LLaVA-7B; `use_dora=True` one-flag in PEFT.
7. TildAlice, *LoRA vs DoRA: 7B Training Speed* (Apr 2026) — DoRA +44% time, +300MB, identical post-merge inference.
8. Unsloth repo + Genαi benchmark (Jul 2026) — Unsloth QLoRA-7B ≈ 5–6GB vs stock ≈ 6–9GB; LLaVA path absent → skipped with reason.
9. TinyLLaVA (arXiv:2402.14289) + MobileVLM repo — 3B-class fallback options with HF IDs.
10. bitsandbytes install docs (Windows x86-64 wheels) + DeepSpeed Offload docs + issues #5585/#1249 — Windows-native viable; ZeRO rejected with cause.

---

## §3 — Recipe commands (copy-paste, Windows PowerShell)

Prereqs (once): Python 3.10/3.11, CUDA 12.x (`nvcc --version` matches torch
build), VS2022 C++ tools (only if building bitsandbytes from source — wheels
preferred), ~25GB disk, MVTec AD extracted to `mvtec_anomaly_detection/`.

```powershell
# 0. Install + verify (from repo root C:\Users\prana\OneDrive\Desktop\VLM)
pip install -e ".[train]"
python scripts/verify_install.py
nvidia-smi   # confirm ~8GB visible, nothing else holding VRAM (close browsers/Docker/desktop overlays)

# 1. Data (once; anomaly fraction MUST match the config's 0.65)
python scripts/prepare_data.py --dataset-root mvtec_anomaly_detection `
  --output-train data/mvtec_train.json --output-test data/mvtec_test.json `
  --anomaly-train-fraction 0.65
python scripts/make_eval_split.py --input data/mvtec_train.json --output eval_test.json   # if harness split needed
# grounding subset already exists: mvtec_grounding_200.json (200 boxes) — do NOT rebuild

# 2. ARM B — QLoRA r=16 (PRIMARY, fits 8GB). Run first, run fully.
python scripts/train.py configs/rtx5070_8gb.yaml
# with WandB:
python scripts/train.py configs/rtx5070_8gb.yaml training.report_to=wandb
# final best-F1 extension (5 epochs, same as v3):
python scripts/train.py configs/rtx5070_8gb.yaml training.num_train_epochs=5

# 3. ARM C — QDoRA r=16 (only after Arm B reproduces; needs peft>=0.9 + ONE-LINE patch:
#    src/vlm_defect/model.py → LoraConfig(..., use_dora=cfg["lora"].get("use_dora", False)))
pip show peft   # confirm >=0.9
python scripts/train.py configs/rtx5070_8gb.yaml training.output_dir=checkpoints/llava-mvtec-qdora-r16-rtx5070
# (+ set use_dora: true in a copy of the config once model.py honours it)

# 4. ARM A — fp16 LoRA (NO QUANT). ⚠️ DOES NOT FIT 8GB — Colab T4/A100 only:
python scripts/train.py configs/rtx5070_8gb.yaml quantization.bits=16 `
  training.output_dir=checkpoints/llava-mvtec-lora-r16-rtx5070 `
  training.per_device_train_batch_size=1 training.gradient_accumulation_steps=16
```

`make train` / `make eval` still work but are pinned to `configs/local_8gb.yaml`;
the commands above target the new profile explicitly (no Makefile change needed).

---

## §4 — Expected VRAM curve per arm (RTX 5070 8GB, bs=1, seq≈730, ckpt ON)

Measure yours with `nvidia-smi -l 1` (reserved) and
`torch.cuda.max_memory_allocated()/1e9` after step 1 (true peak — load-time
readings understate the backward pass by ~2×).

| Phase | Arm B QLoRA r16 (primary) | Arm C QDoRA r16 | Arm A fp16-LoRA (≠8GB) |
|---|---|---|---|
| CUDA context + processor | ~0.6GB | ~0.6GB | ~0.6GB |
| Base weights resident | ~3.9GB (NF4) | ~3.9GB (NF4) | ~14GB (fp16) — **OOM at load** |
| Adapters + grads + AdamW states | ~0.5GB | ~0.6GB (+magnitude vecs) | ~0.5GB |
| Activations, steady-state train | ~1.5–2.0GB | ~1.6–2.2GB | — |
| **Train peak (reserved)** | **~6.5–7.5GB ✅** | **~7.0–8.0GB ⚠️ borderline** | **OOM ❌** |
| Eval spike (every 100 steps; compact 2-logit head keeps it O(B×2)) | +0.3–0.5GB, brief | +0.3–0.5GB | — |
| Checkpoint save (every 100 steps; adapter-only ~100–150MB) | dip, not spike | dip | — |

Curve shape to expect on `nvidia-smi`: fast ramp to ~5GB at load → sawtooth
6.5–7.5GB during train micro-steps → brief +0.5GB tooth at each eval → dip at
each save → repeat. If the sawtooth touches 7.9GB: first halve nothing — set
`dataloader_num_workers=0` (already default here), close VRAM squatters, then
`training.per_device_eval_batch_size=1` (already 1), then reduce
`model_max_length` 2048→1024, then LoRA r 16→8. Do NOT disable grad-ckpt.

---

## §5 — Checkpointing / resume on Windows

- Saves: every 100 steps, adapter-only + tokenizer + `trainer_state.json`,
  keep-last-3 (`save_total_limit: 3`), best-by-F1 retained
  (`load_best_model_at_end`). Expect ~100–150MB per checkpoint (DoRA ~10% more).
- Resume after crash / preemption / laptop sleep (auto-finds latest):
  `python scripts/train.py configs/rtx5070_8gb.yaml --resume`
- Resume a specific checkpoint:
  `python scripts/train.py configs/rtx5070_8gb.yaml --resume checkpoints/llava-mvtec-qlora-rtx5070/checkpoint-700`
- Windows notes: run PowerShell as normal user (not Admin needed); keep the
  repo on NTFS (not OneDrive-on-demand — checkpoints must be local files, or
  sync races corrupt `trainer_state.json`); disable sleep during training
  (`powercfg`), and re-run `nvidia-smi` after any resume to confirm the process
  actually holds the GPU (Windows TDR can silently move it to CPU after long
  idle — symptom: 10× slower steps with 0% GPU util).

---

## §6 — Eval after EACH arm (same order every time)

Run from repo root. `<CKPT>` = best checkpoint dir of the arm
(e.g. `checkpoints/llava-mvtec-qlora-rtx5070/checkpoint-700` — check
`trainer_state.json` `best_model_checkpoint`).

```powershell
# 6a. Harness metrics: text-acc + per-class AUROC + rationale (0-3) + grounding proxy.
#     --mock first (pipeline smoke test, ~seconds), then real preds.
python scripts/eval_mvtec.py --eval-file eval_test.json --mock --output eval_outputs/results_mock_<arm>.json
# <produce eval_outputs/preds_<arm>.json: one {id, response} or {id, label, rationale, bbox_1000} per eval id>
python scripts/eval_mvtec.py --eval-file eval_test.json --preds eval_outputs/preds_<arm>.json `
  --grounding-file mvtec_grounding_200.json --output eval_outputs/results_<arm>.json

# 6b. Threshold sweep + per-category optima (single inference pass, reused for all thresholds):
python scripts/evaluate.py <CKPT> configs/rtx5070_8gb.yaml --sweep-threshold
# then the scored run with learnt thresholds:
python scripts/evaluate.py <CKPT> configs/rtx5070_8gb.yaml --threshold 0.25 --log-failures eval_outputs/failures_<arm>.json
# TTA (2× cost — restrict to hard cats first):
python scripts/evaluate.py <CKPT> configs/rtx5070_8gb.yaml --tta --tta-categories transistor toothbrush screw capsule --sweep-threshold
```

Record per arm: text-acc (normal/defective), mean per-class AUROC, rationale
mean (0–3), grounding detection-rate @IoU≥0.5 + mean IoU (labelled "box-level
PRO proxy" — never as PRO), F1/recall/precision + ROC-AUC from `evaluate.py`,
peak VRAM (`nvidia-smi`), wall-clock. Paste into the ablation table in
`scripts/run_ablation.md` and `REPORT_template.md`.

---

## §7 — Time estimates (RTX 5070 laptop, Arm B defaults, 3 epochs)

Assumptions: ~4 000 train samples/epoch after 65%-anomaly fold-in and 10% val
hold-out (250 optimizer steps/epoch @ eff-batch 16), ~2–4 micro-steps/sec for
7B QLoRA+ckpt on Blackwell-mobile, eval every 100 opt-steps over ~400 val
samples (compact head, minutes each).

| Job | Estimate |
|---|---|
| Arm B train (3 epochs, ~750 opt steps + 7 evals + saves) | **~1–1.5 h** |
| Arm B extension to 5 epochs (override) | **~1.5–2.5 h** total |
| Arm C train (same steps, +~44% norm overhead) | **~1.5–2 h** |
| `eval_mvtec.py` harness per arm (no generation) | minutes |
| `evaluate.py` full test pass (~1.7k samples × generate-32) | **~1–2 h**; TTA doubles it — restrict categories first |
| Merge LoRA + push to Hub (`push_to_hub.py --device cpu`, ~14GB RAM) | ~15–30 min (mostly download/upload) |

If micro-step rate is <1/sec: confirm GPU is actually used (`nvidia-smi`,
bf16 active in log), `dataloader_num_workers=0`, no CPU-offload flags, and no
thermal throttle (laptop on power, fans max). Recalibrate the table with your
first 50 steps: `ETA ≈ total_micro_steps / measured_micro_steps_per_sec`.

---

## §8 — If 7B OOMs anyway (triage order — cheapest first)

1. `nvidia-smi`: kill VRAM squatters (browser, VS Code GPU accel, Docker).
2. Confirm grad-ckpt actually ON in log (`gradient_checkpointing: true` echoes
   via `prepare_model_for_kbit_training`); confirm bf16 (not fp16+overflow).
3. Apply the paged-optimizer one-liner (§2.5) — absorbs eval spikes.
4. `model_max_length` 2048→1024; LoRA r 16→8 (keep all 7 target modules —
   dropping MLP layers costs more quality than dropping rank).
5. `gradient_accumulation_steps` 16→32 AND `eval_steps/save_steps` 100→200
   (fewer eval spikes per epoch; same effective batch math is unchanged —
   note it in REPORT).
6. 3B fallback (different training path, unproven MVTec parity — new experiment,
   not a config flip): `tinyllava/TinyLLaVA-Phi-2-SigLIP-3.1B` or
   `mtgv/MobileVLM_V2-3B` via TinyLLaVA-Factory LoRA/QLoRA recipe; expected
   ~4–5GB peak. Record as its own REPORT arm, never blended into the 7B table.
