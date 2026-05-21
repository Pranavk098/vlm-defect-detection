<div align="center">

# VLM Manufacturing Defect Detection

**Fine-tune LLaVA-1.5-7B on MVTec AD using QLoRA to detect and describe industrial defects**

[![CI](https://github.com/Pranavk098/vlm-defect-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/Pranavk098/vlm-defect-detection/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Model-Hub-yellow)](https://huggingface.co/Pranavk098/llava-mvtec-defect-detection)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Pranavk098/vlm-defect-detection/blob/main/notebooks/LLaVA_Train_Colab.ipynb)

</div>

---

Given a product image and its category, the model answers:
> *"Yes, there is a **broken_large** anomaly."* — or — *"No."*

It runs on **8 GB VRAM** via 4-bit QLoRA and achieves **F1 = 0.887 / ROC-AUC = 0.896** across all 15 MVTec AD categories. The repository documents a full three-version iterative fine-tuning cycle — diagnosis, ablation, and targeted fixes — not just a single training run.

---

## Results at a Glance

| Metric | v1 | v2 | **v3 (current)** |
|--------|----|----|------------------|
| F1 | — | 0.737 | **0.887** |
| Recall | — | 0.585 | **0.940** |
| Precision | — | 0.997 | **0.839** |
| ROC-AUC | ~0.65* | — | **0.896** |
| Accuracy | — | — | 0.860 |

*Evaluated on 1,115 samples across all 15 MVTec AD categories. \*Zero-shot baseline estimate.*

**Best per-category F1** (with per-category thresholds):
`leather=1.000 · bottle=0.955 · carpet=0.978 · grid=0.951`

**Defect-type naming**: raw exact-match 6.8% → constrained-vocab **22.9%** (3.4× via post-hoc fuzzy remapping)

---

## The Engineering Story: v1 → v2 → v3

The key insight is **systematic diagnosis and targeted fixes**, not random hyperparameter search.

### Why v2 failed

`v2` used `no_token_weight=2.7` — upweighting the "No" token to reduce false positives. It backfired:

```
v2 results (threshold=0.70):
  TP=379  FP=1  TN=466  FN=269
  Precision=0.997  Recall=0.585  F1=0.737   ← 269 false negatives
```

Root cause: penalising "No" made the model avoid predicting it — but only when highly confident. The optimal threshold had silently shifted to 0.30, and the hardest categories (`transistor F1=0.095`, `screw=0.123`) collapsed entirely.

### v3 targeted fixes

| Change | Motivation | Result |
|--------|-----------|--------|
| `yes_token_weight=2.0`, `no_token_weight=1.0` | Inverted weighting — penalise missed defects, not false positives | Recall 0.585 → **0.943** |
| LoRA targets: add `k_proj o_proj gate_proj up_proj down_proj` | MLP layers carry the CLIP→LLM fusion pathway; attention-only adapters can't reach them | Stronger visual reasoning |
| LoRA rank r=8 → **r=16** | More adapter capacity for 7 target module groups | Better fit on hard categories |
| `transistor=4× screw=4× capsule=3× pill=2.5×` category weights | Hard categories need more gradient signal | Hard-category F1 significantly improved |
| `anomaly_train_fraction: 0.50 → 0.65` | More defect examples during training | Better anomaly distribution |
| `metric_for_best_model: eval_loss → f1` | eval_loss doesn't correlate with F1 at deployment threshold | Best checkpoint selected correctly |
| Per-category threshold sweep | Global threshold is suboptimal for every category | 5–15 pt per-category F1 gains |

---

## Requirements

| | Minimum | Recommended |
|--|---------|-------------|
| **GPU VRAM** | 8 GB | 16 GB+ |
| **Python** | 3.10 | 3.11 |
| **CUDA** | 11.8 | 12.1 |
| **Disk** | 25 GB | 50 GB |

> **No GPU?** Use Google Colab (T4/A100). Open the notebook with the badge above.
>
> **Windows users:** `bitsandbytes` requires WSL2 or Docker — it has no native Windows build.

---

## Quick Start

### 1. Clone & install

```bash
git clone https://github.com/Pranavk098/vlm-defect-detection.git
cd vlm-defect-detection
pip install -e ".[train]"   # transformers, peft, bitsandbytes, wandb
make verify                 # sanity-check all imports
```

### 2. Download the dataset

Register and download [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad), then extract it into the project root:

```
vlm-defect-detection/
└── mvtec_anomaly_detection/
    ├── bottle/  ├── cable/  ├── capsule/  ...  (15 categories)
```

### 3. Prepare → Train → Evaluate

```bash
make prepare   # builds data/mvtec_{train,test}.json (~5,000 SFT records)
make train     # 5 epochs, effective batch=16, WandB logging
               # override any key: make train OVERRIDE="lora.r=32"

# Full evaluation with per-category thresholds + TTA on hard categories:
vlm-eval checkpoints/llava-mvtec-lora-v3/checkpoint-500 configs/local_8gb.yaml \
    --sweep-threshold --tta --tta-categories transistor toothbrush \
    --log-failures failures.json
```

### 4. Run the Gradio demo

```bash
vlm-app --checkpoint checkpoints/llava-mvtec-lora-v3/checkpoint-500 \
        --config configs/local_8gb.yaml
```

The UI includes a **category dropdown**, **threshold slider** (auto-set to the sweep-optimal value), and a **CLIP attention map overlay** showing where the vision encoder attends.

### 5. Compare checkpoints across versions

```bash
python scripts/compare_checkpoints.py \
    checkpoints/llava-mvtec-lora \
    checkpoints/llava-mvtec-lora-v2 \
    checkpoints/llava-mvtec-lora-v3 \
    --compare-versions --defect-stats
```

### Docker

```bash
make docker-build
make docker-train OVERRIDE="training.report_to=wandb"
```

> See [SETUP.md](SETUP.md) for the full step-by-step walkthrough including troubleshooting.

---

## Run Inference

```python
import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

repo_id = "Pranavk098/llava-mvtec-defect-detection"
processor = AutoProcessor.from_pretrained(repo_id)
model = LlavaForConditionalGeneration.from_pretrained(
    repo_id, torch_dtype=torch.bfloat16, device_map="auto"
).eval()

image    = Image.open("path/to/image.png").convert("RGB")
category = "bottle"   # must match the product in the image

prompt = (
    f"USER: <image>\n"
    f"Is there any anomaly in this {category} image? "
    f"If yes, say 'Yes, there is a <defect_type> anomaly.' "
    f"If no, say 'No.' ASSISTANT:"
)

inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
with torch.inference_mode():
    out = model.generate(**inputs, max_new_tokens=64, do_sample=False)

answer = processor.tokenizer.decode(
    out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True
)
print(answer)
# → "Yes, there is a broken_large anomaly."  or  "No."
```

**Getting a confidence score** (for threshold-based classification):

```python
import torch.nn.functional as F

with torch.inference_mode():
    logits = model(**inputs).logits
    next_token_logits = logits[0, -1, :]

yes_id = processor.tokenizer.encode("Yes", add_special_tokens=False)[0]
no_id  = processor.tokenizer.encode("No",  add_special_tokens=False)[0]
probs  = F.softmax(next_token_logits[[yes_id, no_id]].float(), dim=0)
prob_yes = probs[0].item()

# Apply per-category threshold from the v3 sweep:
THRESHOLDS = {
    "bottle": 0.20, "cable": 0.25, "capsule": 0.30, "carpet": 0.15,
    "grid": 0.15, "hazelnut": 0.20, "leather": 0.40, "metal_nut": 0.50,
    "pill": 0.25, "screw": 0.40, "tile": 0.50, "toothbrush": 0.80,
    "transistor": 0.40, "wood": 0.30, "zipper": 0.40,
}
is_anomaly = prob_yes > THRESHOLDS.get(category, 0.25)
```

---

## Key Techniques

| Technique | File | Purpose |
|-----------|------|---------|
| 4-bit NF4 QLoRA | `model.py` | 7B model fits in 8 GB VRAM |
| LoRA on attention + MLP (r=16) | `configs/local_8gb.yaml` | Adapts the CLIP→LLM fusion pathway |
| Token-level weighted CE loss | `trainer.py` | Controls recall/precision at the loss level |
| F1-based checkpoint selection | `configs/` | Selects the checkpoint that matters at deployment |
| Category-weighted sampler | `trainer.py + data.py` | Per-category oversampling for hard categories |
| Category-aware prompts | `data.py` | Model knows which product it is inspecting |
| Centre-crop augmentation | `data.py` | Zooms small-defect categories for the 336px encoder |
| Per-category threshold sweep | `evaluate.py` | Post-hoc calibration without retraining |
| Test-Time Augmentation | `evaluate.py --tta` | Horizontal flip averaging for hard categories |
| CLIP attention map overlay | `app.py` | Visual explanation of model attention |

---

## Known Limitation: Defect Name Collapse

The model reliably detects **whether** a defect exists (F1=0.887), but struggles to name **which** defect type it is.

```
pred=scratch:   452 / 615 anomaly predictions  (73.5%)  ← universal fallback
pred=None:      163 / 615                       (26.5%)  ← false negatives
```

**Root cause:** `yes_token_weight` targets only the first `Yes`/`No` token. All subsequent tokens are trained with standard CE loss, so the model memorised `"scratch"` — the most frequent defect word across training categories.

**Post-hoc fix (implemented):** `_constrained_defect_name(pred, category)` in `evaluate.py` remaps free-form output to the nearest ground-truth label via fuzzy matching → exact-match 6.8% → **22.9%** (3.4×).

**Proper fix (requires retraining):** Apply CE loss over the full generated sequence, or use constrained beam search with a per-category token whitelist at generation time.

---

## Per-Category Results

| Category | n | Accuracy | Recall | F1 | Threshold |
|----------|---|----------|--------|----|-----------|
| leather | 80 | 1.000 | 1.000 | **1.000** | 0.40 |
| carpet | 75 | 0.973 | 0.957 | **0.978** | 0.15 |
| grid | 51 | 0.941 | 0.967 | **0.951** | 0.15 |
| bottle | 52 | 0.942 | 1.000 | **0.955** | 0.20 |
| tile | 76 | 0.921 | 0.907 | 0.929 | 0.50 |
| wood | 50 | 0.920 | 1.000 | 0.939 | 0.25 |
| hazelnut | 76 | 0.908 | 0.944 | 0.907 | 0.20 |
| pill | 99 | 0.838 | 0.918 | 0.893 | 0.25 |
| metal_nut | 70 | 0.829 | 0.958 | 0.885 | 0.50 |
| toothbrush | 27 | 0.852 | 0.933 | 0.875 | 0.80 |
| cable | 105 | 0.867 | 0.915 | 0.860 | 0.25 |
| capsule | 79 | 0.797 | 0.911 | 0.864 | 0.30 |
| zipper | 93 | 0.785 | 0.951 | 0.853 | 0.40 |
| screw | 102 | 0.706 | 0.934 | 0.792 | 0.40 |
| transistor | 80 | 0.762 | 0.750 | 0.612 | 0.40 |

> **Transistor** is the hardest category: F1=0.612. Bent leads and misplaced components require spatial orientation understanding that the CLIP ViT-L/14 encoder struggles with.

---

## Project Structure

```
├── configs/
│   └── local_8gb.yaml          # All training hyper-parameters (v3)
├── data/                       # Generated by make prepare (gitignored)
│   ├── mvtec_train.json        # ~5,000 SFT records
│   └── mvtec_test.json         # ~1,115 held-out evaluation records
├── notebooks/
│   ├── LLaVA_Train_Colab.ipynb
│   └── VLM_LLaVA_Colab_Training.ipynb
├── scripts/
│   ├── prepare_data.py         # Build SFT data from MVTec AD
│   ├── train.py                # Launch fine-tuning (reads YAML config)
│   ├── evaluate.py             # Full evaluation: metrics, sweep, TTA, failures
│   ├── compare_checkpoints.py  # Cross-version checkpoint comparison
│   ├── zero_shot_baseline.py   # Evaluate base model without adapter
│   └── push_to_hub.py          # Merge LoRA + push to HuggingFace Hub
├── src/vlm_defect/
│   ├── model.py                # 4-bit QLoRA model loading
│   ├── data.py                 # MVTecDataset, centre-crop, weighted sampler
│   ├── trainer.py              # WeightedCETrainer (custom loss + prediction_step)
│   ├── evaluate.py             # Evaluation engine (TTA, per-category sweep)
│   └── app.py                  # Gradio demo
├── tests/                      # pytest suite
├── .github/workflows/ci.yml    # Lint, import check, smoke test
├── Dockerfile
├── Makefile
├── SETUP.md                    # Full training walkthrough & troubleshooting
└── pyproject.toml
```

---

## Configuration

All hyper-parameters live in `configs/local_8gb.yaml`. Any key can be overridden at the CLI:

```bash
python scripts/train.py configs/local_8gb.yaml training.num_train_epochs=5 lora.r=32
```

Key v3 settings:

```yaml
lora:
  r: 16              # doubled from v1's r=8
  alpha: 32
  target_modules: [q_proj, v_proj, k_proj, o_proj, gate_proj, up_proj, down_proj]

class_balance:
  yes_token_weight: 2.0  # upweight "Yes" to improve recall

category_weights:        # oversample hard categories
  transistor: 4.0
  screw:      4.0
  capsule:    3.0
  pill:       2.5

training:
  metric_for_best_model: "f1"   # not eval_loss
  anomaly_train_fraction: 0.65
```

---

## Tech Stack

| Component | Choice |
|-----------|--------|
| Base model | LLaVA-1.5-7B (`llava-hf/llava-1.5-7b-hf`) |
| Dataset | MVTec Anomaly Detection (15 categories, 1,115 test samples) |
| Quantization | 4-bit NF4 QLoRA (`bitsandbytes`) |
| LoRA | PEFT — full attention + MLP layers, r=16 |
| Training framework | HuggingFace Transformers + custom `WeightedCETrainer` |
| Experiment tracking | WandB (7 logged runs) |
| Demo | Gradio (category dropdown, threshold slider, attention map) |
| Environment | Google Colab / Docker / local 8 GB VRAM |

---

## Contributing

Contributions are welcome. Please open an issue first to discuss what you'd like to change, then submit a pull request.

Areas where contributions would be especially valuable:
- Full-sequence CE loss for defect type naming (see [Known Limitation](#known-limitation-defect-name-collapse))
- Constrained beam search with per-category token whitelist
- Support for additional anomaly detection datasets (BTAD, VisA)
- Pixel-level segmentation output

Run the test suite before submitting:

```bash
pip install -e ".[dev]"
pytest tests/
```

---

## Acknowledgements

- [LLaVA](https://llava-vl.github.io/) — Liu et al., *Visual Instruction Tuning*, 2023
- [QLoRA](https://arxiv.org/abs/2305.14314) — Dettmers et al., 2023
- [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) — Bergmann et al., CVPR 2019

---

## Citation

```bibtex
@inproceedings{bergmann2019mvtec,
  title     = {MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection},
  author    = {Bergmann, Paul and Fauser, Michael and Sattlegger, David and Steger, Carsten},
  booktitle = {CVPR},
  year      = {2019},
}

@misc{liu2023llava,
  title  = {Visual Instruction Tuning},
  author = {Liu, Haotian and Li, Chunyuan and Wu, Qingyang and Lee, Yong Jae},
  year   = {2023},
  eprint = {2304.08485},
}

@misc{dettmers2023qlora,
  title  = {QLoRA: Efficient Finetuning of Quantized LLMs},
  author = {Dettmers, Tim and Pagnoni, Artidoro and Fansi, Ari and Zettlemoyer, Luke},
  year   = {2023},
  eprint = {2305.14314},
}
```

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
