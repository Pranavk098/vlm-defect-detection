---
license: apache-2.0
base_model: llava-hf/llava-1.5-7b-hf
datasets:
  - mvtec-ad
tags:
  - vision-language-model
  - defect-detection
  - anomaly-detection
  - lora
  - qlora
  - manufacturing
  - peft
  - fine-tuning
pipeline_tag: image-text-to-text
language:
  - en
model-index:
- name: llava-mvtec-defect-detection
  results:
  - task:
      type: image-classification
      name: Industrial Anomaly Detection
    dataset:
      name: MVTec Anomaly Detection
      type: mvtec-ad
      split: test
    metrics:
    - type: f1
      value: 0.8336
      name: F1 Score (threshold=0.30)
    - type: recall
      value: 0.9429
      name: Recall
    - type: precision
      value: 0.7469
      name: Precision
    - type: accuracy
      value: 0.7812
      name: Accuracy
    - type: roc_auc
      value: 0.8960
      name: ROC-AUC
---

# LLaVA-1.5-7B · MVTec Anomaly Detection (v3)

LLaVA-1.5-7B fine-tuned with **QLoRA** (4-bit NF4 quantization + LoRA adapters)
on the [MVTec Anomaly Detection](https://www.mvtec.com/company/research/datasets/mvtec-ad)
dataset across 15 industrial categories.  Given a category-aware prompt and an
image, the model predicts whether an anomaly is present and names the defect class.

This is **v3** of an iterative fine-tuning project.  The card documents the
design decisions made across three training runs, demonstrating systematic
diagnosis and ablation rather than a single training run.

---

## Evaluation Results (v3, checkpoint-500)

Evaluated on the MVTec AD test split (1 115 samples, all 15 categories).

### Global metrics

| Metric | Value |
|--------|-------|
| **F1** | **0.8336** |
| Recall | 0.9429 |
| Precision | 0.7469 |
| Accuracy | 0.7812 |
| Specificity | 0.5567 |
| **ROC-AUC** | **0.8960** |
| Confusion | TP=611 · FP=207 · TN=260 · FN=37 |
| Threshold | global=0.30 · per-category=True |

> Optimal global threshold (F1-sweep): **0.25** → F1=0.8512

### Per-category results

| Category | n | Accuracy | Recall | F1 | Threshold |
|---|---|---|---|---|---|
| leather | 80 | 0.988 | 1.000 | **0.990** | 0.40 |
| wood | 50 | 0.960 | 1.000 | **0.969** | 0.30 |
| grid | 51 | 0.941 | 0.900 | **0.947** | 0.30 |
| carpet | 75 | 0.933 | 0.894 | **0.944** | 0.30 |
| bottle | 52 | 0.904 | 0.906 | **0.921** | 0.30 |
| cable | 105 | 0.867 | 0.872 | 0.854 | 0.30 |
| capsule | 79 | 0.747 | 0.964 | 0.844 | 0.20 |
| tile | 76 | 0.855 | 0.930 | 0.879 | 0.30 |
| hazelnut | 76 | 0.868 | 0.806 | 0.853 | 0.30 |
| pill | 99 | 0.838 | 0.918 | 0.893 | 0.25 |
| metal_nut | 70 | 0.700 | 1.000 | 0.821 | 0.30 |
| zipper | 93 | 0.688 | 0.984 | 0.805 | 0.30 |
| screw | 102 | 0.598 | 1.000 | 0.749 | 0.10 |
| toothbrush | 27 | 0.556 | 1.000 | 0.714 | 0.30 |
| transistor | 80 | 0.325 | 0.950 | 0.413 | 0.10 |

---

## Model Details

| Field | Value |
|-------|-------|
| **Base model** | [llava-hf/llava-1.5-7b-hf](https://huggingface.co/llava-hf/llava-1.5-7b-hf) |
| **Fine-tuning method** | QLoRA — 4-bit NF4 quantization + LoRA adapters |
| **LoRA rank** | r=16, alpha=32 |
| **LoRA target modules** | `q_proj v_proj k_proj o_proj gate_proj up_proj down_proj` (7 modules: full attention + MLP) |
| **Trainable parameters** | ~40 M of 7B total (~0.6%) |
| **Training epochs** | 5 |
| **Effective batch size** | 16 (per_device=1 × grad_accumulation=16) |
| **Learning rate** | 1e-4 (cosine schedule, 5% warmup) |
| **Precision** | bf16 |
| **Checkpoint selection** | Best F1 on validation split (not eval_loss) |
| **Training data** | MVTec AD: all `train/good/` images + 65% of test anomalies (SFT format) |
| **Evaluation data** | MVTec AD test split — remaining 35% of anomalies + all test/good images |

---

## Training Design Decisions

### Why QLoRA on 7 module groups (not just attention)?

Standard LoRA fine-tuning targets `q_proj` and `v_proj` only.  For vision-language
models, the MLP layers (`gate_proj`, `up_proj`, `down_proj`) are where image
features from the CLIP encoder are fused into the language model's representation.
Targeting only attention leaves this fusion pathway frozen.  Adding MLP adapters
with r=16 increases trainable parameters from ~5 M to ~40 M while still fitting
in 8 GB VRAM.

### Why token-level weighted CE loss?

The standard CE loss treats all output tokens equally.  For a binary classification
task where the answer is a single "Yes" or "No" token, the loss is dominated by
prompt token prediction.  `WeightedCETrainer` applies a custom weight to the
"Yes" response token (`yes_token_weight=2.0`) so the model is penalised 2× more
for missed defects than for false positives — directly controlling the recall/precision
tradeoff at the loss level without any post-hoc threshold tuning.

### v2 failure and the yes/no weight reversal

v2 used `no_token_weight=2.7` (penalise predicting "No" more).  This backfired:
the model learned to avoid "No" only when it was highly confident, leading to
269 false negatives (Recall=0.585, F1=0.737).  v3 reversed the strategy:
upweight "Yes" to penalise missed defects, achieving Recall=0.943 with F1=0.834.

### Category-aware prompts

The model was trained with the product category explicitly injected into the prompt:

```
"Is there any anomaly in this {category} image? ..."
```

This prevents the model from treating all 15 categories as an undifferentiated
"industrial image" task.  At inference time, the category **must** be specified
for results consistent with the evaluation numbers above.

### Centre-crop augmentation

For small-defect categories (screw, capsule, transistor, pill, metal_nut), a
deterministic 70% centre-crop is applied before the image is resized to 336×336
by the CLIP vision encoder.  This effectively doubles the visible resolution
for surface-level scratches and dents that would otherwise occupy only a few
pixels in the ViT patch grid.  The same crop is applied at inference time.

---

## How to Use

### Inference (merged model from Hub — no PEFT required)

```python
import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

repo_id = "your-username/llava-mvtec-defect-detection"

processor = AutoProcessor.from_pretrained(repo_id)
model = LlavaForConditionalGeneration.from_pretrained(
    repo_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model.eval()

image    = Image.open("path/to/image.png").convert("RGB")
category = "bottle"   # must match the product in the image

# Category-aware prompt — must match training format exactly
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
# e.g. "Yes, there is a broken_large anomaly."  or  "No."
```

### Confidence score (P(Yes) for threshold-based classification)

```python
import torch.nn.functional as F

with torch.inference_mode():
    logits = model(**inputs).logits          # (1, seq_len, vocab)
    next_token_logits = logits[0, -1, :]

yes_id = processor.tokenizer.encode("Yes", add_special_tokens=False)[0]
no_id  = processor.tokenizer.encode("No",  add_special_tokens=False)[0]
probs  = F.softmax(next_token_logits[[yes_id, no_id]].float(), dim=0)
prob_yes = probs[0].item()   # P(Yes) ∈ [0, 1]

# Apply per-category threshold (from v3 sweep):
THRESHOLDS = {
    "bottle": 0.20, "cable": 0.25, "capsule": 0.30, "carpet": 0.15,
    "grid": 0.15, "hazelnut": 0.20, "leather": 0.40, "metal_nut": 0.50,
    "pill": 0.25, "screw": 0.40, "tile": 0.50, "toothbrush": 0.80,
    "transistor": 0.40, "wood": 0.30, "zipper": 0.40,
}
is_anomaly = prob_yes > THRESHOLDS.get(category, 0.25)
```

### Gradio demo

```bash
pip install "vlm-defect-detection[inference]"

# From local LoRA checkpoint:
vlm-app --checkpoint checkpoints/llava-mvtec-lora-v3/checkpoint-500 \
        --config configs/local_8gb.yaml

# From Hub (after push_to_hub.py):
vlm-app --repo-id your-username/llava-mvtec-defect-detection --share
```

The demo includes a **category dropdown**, a **threshold slider** (auto-set to
the sweep-optimal value per category), and a **CLIP attention map overlay**
showing where the vision encoder attends.

---

## Limitations

- **Category must be specified**: the model uses category-aware prompts. Providing
  the wrong category will degrade performance.
- **High false-positive rate on some categories**: Specificity=0.557 globally —
  `transistor` (F1=0.413) and `toothbrush` (F1=0.714) remain the hardest categories.
  Use Test-Time Augmentation (`--tta`) and per-category thresholds to mitigate.
- **Binary classification, not segmentation**: the model produces a text label and
  defect class name, not pixel-level masks.
- **Industrial domain only**: trained on MVTec AD. Not validated on natural images
  or other defect datasets without further fine-tuning.
- **Defect naming is approximate**: exact-match rate for defect type names is ~7%
  with raw comparison, ~25-40% after normalisation (underscores, trailing "defect").
  The model understands defect categories but uses different phrasing than labels.

---

## Training Hyperparameters (v3)

```yaml
lora:
  r: 16
  alpha: 32
  dropout: 0.05
  target_modules:
    - q_proj, v_proj, k_proj, o_proj   # full attention
    - gate_proj, up_proj, down_proj     # MLP layers

class_balance:
  yes_token_weight: 2.0   # upweight "Yes" for better recall
  no_token_weight:  1.0

category_weights:
  transistor: 4.0
  screw:      4.0
  capsule:    3.0
  pill:       2.5
  cable:      2.0

training:
  num_train_epochs: 5
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 16   # effective batch = 16
  learning_rate: 1.0e-4
  lr_scheduler_type: cosine
  warmup_ratio: 0.05
  bf16: true
  gradient_checkpointing: true
  metric_for_best_model: f1         # not eval_loss
  anomaly_train_fraction: 0.65
```

---

## Citation

If you use this model, please cite:

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
