"""Training orchestration using the native HuggingFace Trainer.

LlavaForConditionalGeneration has been in HuggingFace transformers natively
since v4.36 — no external LLaVA repo clone is required.

Usage:
    vlm-train configs/local_8gb.yaml                     # installed entrypoint
    python scripts/train.py configs/local_8gb.yaml        # via script
    make train                                            # via Makefile

    # CLI key=value overrides:
    vlm-train configs/local_8gb.yaml training.report_to=wandb

Improvements (v3):
  - yes_token_weight: upweights "Yes" response tokens in the loss to improve
    recall (replaces the v2 no_token_weight which was hurting recall).
  - memory-efficient prediction_step: instead of materialising full [B, T, V]
    logits during eval (which can OOM at large T), we return only the 2-element
    [yes_logit, no_logit] at the first response token per sample. This makes
    eval O(B × 2) rather than O(B × T × V) in memory.
  - compute_metrics / F1-based checkpoint selection: a compute_metrics callback
    decodes the compact eval predictions into binary F1/recall/precision so
    that load_best_model_at_end selects the checkpoint with the highest F1
    rather than lowest eval_loss.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from transformers import Trainer, TrainerCallback, TrainingArguments, EvalPrediction

from vlm_defect.data import collate_fn, load_split, make_weighted_sampler
from vlm_defect.model import load_model_and_processor


# ---------------------------------------------------------------------------
# WandB loss display fix
# ---------------------------------------------------------------------------

class NormalizedLossCallback(TrainerCallback):
    """Fix WandB train/loss display when using gradient accumulation.

    Root cause: HuggingFace Trainer accumulates raw ``compute_loss`` return
    values into ``tr_loss`` and then logs ``tr_loss / n_optimizer_steps``.
    With ``gradient_accumulation_steps=N``, each optimizer step runs N
    micro-batches, so ``tr_loss`` grows N× per optimizer step — making WandB
    show ``N × actual_loss`` (e.g. 16 × 3.5 ≈ 56) instead of the true ~3.5.

    This callback divides the logged ``loss`` value by
    ``gradient_accumulation_steps`` *before* it is sent to WandB/console.
    It is purely cosmetic — it does not affect gradients or model weights.
    """

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and "loss" in logs and args.gradient_accumulation_steps > 1:
            logs["loss"] = round(
                logs["loss"] / args.gradient_accumulation_steps, 6
            )


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def apply_overrides(cfg: dict, overrides: list[str]) -> None:
    """Apply dotted key=value overrides, e.g. ``training.report_to=wandb``."""
    for override in overrides:
        key, _, value = override.partition("=")
        parts = key.split(".")
        node = cfg
        for part in parts[:-1]:
            node = node[part]
        if value.lower() in ("true", "false"):
            value = value.lower() == "true"
        else:
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    pass
        node[parts[-1]] = value


# ---------------------------------------------------------------------------
# Compute-metrics factory for F1-based checkpoint selection
# ---------------------------------------------------------------------------

def make_compute_metrics(threshold: float = 0.3):
    """Return a compute_metrics function for the HuggingFace Trainer.

    The function receives compact eval predictions — each sample contributes a
    2-element ``[yes_logit, no_logit]`` vector and a binary ground-truth label
    (1 = anomaly, 0 = normal) — and returns binary classification metrics
    (f1, recall, precision, accuracy).

    These compact predictions are produced by
    ``WeightedCETrainer.prediction_step``.  The ground-truth labels are encoded
    as integers (1/0) rather than raw token IDs to avoid LLaMA SentencePiece
    space-prefix tokenisation issues (where ``encode("Yes")`` may return a
    different ID than the contextual "▁Yes" token that actually appears in the
    label tensor).

    Args:
        threshold: P(Yes) threshold for the binary decision (default 0.3).

    Returns:
        A callable ``(EvalPrediction) → dict`` compatible with
        ``TrainingArguments.compute_metrics``.
    """
    def compute_metrics(eval_pred: EvalPrediction) -> dict[str, float]:
        pair_logits, first_labels = eval_pred.predictions, eval_pred.label_ids
        # pair_logits: (N, 2) float — [yes_logit, no_logit] per sample
        # first_labels: (N,)  int  — 1 (anomaly) / 0 (normal) / -100 (padding)

        # Ignore padding samples (sentinel value -100)
        valid = first_labels != -100
        if not np.any(valid):
            return {"f1": 0.0, "recall": 0.0, "precision": 0.0, "accuracy": 0.0}

        pair_logits  = pair_logits[valid].astype(np.float64)
        first_labels = first_labels[valid]

        # Softmax over [yes_logit, no_logit] → P(Yes)
        max_logit = pair_logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(pair_logits - max_logit)
        prob_yes = exp_logits[:, 0] / exp_logits.sum(axis=1)

        pred_yes = prob_yes > threshold  # (N,) bool
        true_yes = first_labels == 1     # (N,) bool — 1=anomaly stored by prediction_step

        tp = int(np.sum(true_yes & pred_yes))
        fp = int(np.sum(~true_yes & pred_yes))
        fn = int(np.sum(true_yes & ~pred_yes))
        tn = int(np.sum(~true_yes & ~pred_yes))
        total = tp + fp + fn + tn

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall    = tp / (tp + fn) if (tp + fn) else 0.0
        f1        = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) else 0.0
        )
        accuracy  = (tp + tn) / total if total else 0.0

        return {
            "f1":        round(f1, 4),
            "recall":    round(recall, 4),
            "precision": round(precision, 4),
            "accuracy":  round(accuracy, 4),
        }

    return compute_metrics


# ---------------------------------------------------------------------------
# WeightedCETrainer
# ---------------------------------------------------------------------------

class WeightedCETrainer(Trainer):
    """HuggingFace Trainer subclass with three improvements over the baseline.

    1. **Token-level class weighting** (``yes_token_weight`` / ``no_token_weight``):
       The CE loss for every token matching the "Yes" (or "No") token id is
       multiplied by the corresponding weight.  v3 default: yes_token_weight=2.0
       to improve recall; no_token_weight kept for backwards compat (default 1.0).

    2. **Memory-efficient prediction_step**:
       Returns only the 2-element [yes_logit, no_logit] vector at the first
       response position per sample, avoiding O(B × T × V) memory during eval.
       This feeds directly into ``compute_metrics`` for F1 computation.

    3. **Category-weighted data loader** (``weighted_sampler``):
       When a ``WeightedRandomSampler`` is supplied the default DataLoader is
       replaced with one that uses it, oversampling hard categories such as
       *screw*, *capsule*, and *transistor* during training.
    """

    def __init__(
        self,
        *args,
        yes_token_id: Optional[int] = None,
        yes_token_weight: float = 1.0,
        no_token_id: Optional[int] = None,
        no_token_weight: float = 1.0,
        weighted_sampler=None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.yes_token_id      = yes_token_id
        self.yes_token_weight  = yes_token_weight
        self.no_token_id       = no_token_id
        self.no_token_weight   = no_token_weight
        self._weighted_sampler = weighted_sampler

    # ------------------------------------------------------------------
    # Override: weighted cross-entropy loss
    # ------------------------------------------------------------------

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Compute per-token CE loss with optional yes/no token upweighting.

        Labels are popped from *inputs* before the forward pass so that
        LlavaForConditionalGeneration does not compute its own (unweighted)
        loss internally — we compute it ourselves below.
        """
        labels = inputs.pop("labels", None)

        outputs = model(**inputs)
        logits  = outputs.logits  # [B, T, V]

        if labels is None:
            loss = outputs.loss
            inputs["labels"] = labels
            return (loss, outputs) if return_outputs else loss

        # Restore labels so other Trainer hooks can access them if needed
        inputs["labels"] = labels

        # Shift for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()   # [B, T-1, V]
        shift_labels = labels[..., 1:].contiguous()        # [B, T-1]

        loss_fct = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
        per_token_loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )  # [B*(T-1)]

        flat_labels = shift_labels.view(-1)
        valid_mask  = flat_labels != -100

        # Determine whether any token-level weighting is active
        need_weighting = (
            (self.yes_token_id is not None and self.yes_token_weight != 1.0) or
            (self.no_token_id  is not None and self.no_token_weight  != 1.0)
        )

        if need_weighting:
            token_weights = torch.ones_like(per_token_loss)
            # Upweight "Yes" tokens — improves recall
            if self.yes_token_id is not None and self.yes_token_weight != 1.0:
                token_weights[flat_labels == self.yes_token_id] = self.yes_token_weight
            # Upweight "No" tokens — improves precision (kept for backwards compat)
            if self.no_token_id is not None and self.no_token_weight != 1.0:
                token_weights[flat_labels == self.no_token_id] = self.no_token_weight

            weighted_loss = per_token_loss * token_weights * valid_mask.float()
            denom = (token_weights * valid_mask.float()).sum().clamp(min=1.0)
            loss  = weighted_loss.sum() / denom
        else:
            loss = per_token_loss[valid_mask].mean()

        return (loss, outputs) if return_outputs else loss

    # ------------------------------------------------------------------
    # Override: memory-efficient prediction_step
    # ------------------------------------------------------------------

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Return compact (loss, pair_logits, binary_labels) instead of full logits.

        Instead of returning the full [B, T, V] logit tensor (which can require
        tens of GB at T=2048, V=32000), this override extracts only the
        [yes_logit, no_logit] pair at the first non-masked (response) position
        for each sample.  The Trainer stacks these across batches so
        ``compute_metrics`` receives (N, 2) predictions and (N,) label_ids.

        Ground-truth labels are stored as **integers (1=anomaly, 0=normal)**
        rather than raw token IDs.  This avoids LLaMA SentencePiece
        space-prefix tokenisation ambiguity: in context "ASSISTANT: Yes,…" the
        first response token may be "▁Yes" (space-prefixed) whose ID differs
        from ``tokenizer.encode("Yes")[0]``.  Decoding the token to a string
        and checking ``startswith("y")`` is unambiguous.

        Requires ``self.tokenizer`` to be set (pass ``tokenizer=`` to the
        Trainer constructor — handled in ``main()``).

        Falls back to the parent implementation when yes_token_id is not set
        (e.g. when no compute_metrics is provided).
        """
        inputs = self._prepare_inputs(inputs)

        with torch.inference_mode():
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
        loss = loss.detach()

        if prediction_loss_only or self.yes_token_id is None:
            return (loss, None, None)

        logits = outputs.logits   # [B, T, V]
        labels = inputs.get("labels")

        # Tokenizer is required for robust label decoding
        tokenizer = getattr(self, "tokenizer", None) or getattr(self, "processing_class", None)

        B = logits.shape[0]
        pair_logits_list: list[torch.Tensor] = []
        first_labels_list: list[torch.Tensor] = []

        for i in range(B):
            if labels is not None:
                non_masked = (labels[i] != -100).nonzero(as_tuple=True)[0]
            else:
                non_masked = torch.tensor([], dtype=torch.long, device=logits.device)

            if len(non_masked) == 0:
                # No response tokens — use sentinel so compute_metrics skips this sample
                pair_logits_list.append(torch.zeros(2, device=logits.device, dtype=torch.float))
                first_labels_list.append(torch.tensor(-100, dtype=torch.long, device=logits.device))
                continue

            first_pos = non_masked[0].item()
            # logits at (first_pos - 1) predict the token at first_pos
            pred_pos  = max(0, first_pos - 1)
            yes_no_ids = torch.tensor(
                [self.yes_token_id, self.no_token_id],
                device=logits.device,
            )
            pair_logits_list.append(logits[i, pred_pos, yes_no_ids].float())

            # ── Binary ground-truth label ────────────────────────────────────
            # Decode the true first-response token to text and check whether it
            # starts with "y"/"Y" (="Yes") or "n"/"N" (="No").  This is robust
            # against SentencePiece space-prefix variants ("▁Yes" vs "Yes").
            label_tok_id = labels[i, first_pos].item()
            if tokenizer is not None and label_tok_id >= 0:
                token_text = tokenizer.decode([label_tok_id]).strip().lower()
                is_anomaly = 1 if token_text.startswith("y") else 0
            else:
                # Fallback: compare raw token ID (may mis-classify space-prefix variants)
                is_anomaly = 1 if label_tok_id == self.yes_token_id else 0
            first_labels_list.append(
                torch.tensor(is_anomaly, dtype=torch.long, device=logits.device)
            )

        pair_logits  = torch.stack(pair_logits_list)   # [B, 2]
        first_labels = torch.stack(first_labels_list)  # [B]  — values: 1, 0, or -100

        return (loss, pair_logits, first_labels)

    # ------------------------------------------------------------------
    # Override: category-weighted data loader
    # ------------------------------------------------------------------

    def get_train_dataloader(self) -> DataLoader:
        if self._weighted_sampler is None:
            return super().get_train_dataloader()

        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            sampler=self._weighted_sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=True,
            drop_last=self.args.dataloader_drop_last,
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune LLaVA on MVTec AD using HuggingFace Trainer"
    )
    parser.add_argument("config", type=Path, help="Path to YAML config file")
    parser.add_argument(
        "overrides",
        nargs="*",
        metavar="KEY=VALUE",
        help="Dotted key=value overrides, e.g. training.report_to=wandb",
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        const=True,
        default=False,
        metavar="CHECKPOINT_DIR",
        help=(
            "Resume training from a checkpoint. "
            "Use --resume to auto-detect the latest checkpoint in output_dir, "
            "or --resume checkpoints/llava-mvtec-lora-v3/checkpoint-700 to pick one."
        ),
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    apply_overrides(cfg, args.overrides)

    project_dir = Path(__file__).parent.parent.parent.absolute()
    t = cfg["training"]
    d = cfg["data"]

    data_json = project_dir / d["path"]
    if not data_json.exists():
        print(f"[ERROR] Training data not found: {data_json}")
        print("  Run:  make prepare   (or: vlm-prepare)")
        sys.exit(1)

    image_folder = project_dir / d["image_folder"]

    print("[INFO] Loading model and processor...")
    model, processor = load_model_and_processor(cfg)

    print("[INFO] Building train/val split...")
    train_dataset, val_dataset = load_split(
        json_path=data_json,
        image_folder=image_folder,
        processor=processor,
        model_max_length=t["model_max_length"],
        val_fraction=d.get("val_fraction", 0.1),
    )

    # ── Class-balance: resolve token ids ─────────────────────────────────────
    cb_cfg = cfg.get("class_balance", {})

    # "Yes" token upweighting (v3: primary knob for improving recall)
    yes_token_weight = float(cb_cfg.get("yes_token_weight", 1.0))
    yes_token_id     = None
    if yes_token_weight != 1.0:
        encoded = processor.tokenizer.encode("Yes", add_special_tokens=False)
        yes_token_id = encoded[0] if encoded else None
        print(
            f"[INFO] Class weighting: 'Yes' token (id={yes_token_id}) "
            f"upweighted ×{yes_token_weight:.2f}"
        )

    # "No" token upweighting (v2 legacy; default 1.0 = disabled in v3)
    no_token_weight = float(cb_cfg.get("no_token_weight", 1.0))
    no_token_id     = None
    if no_token_weight != 1.0:
        encoded = processor.tokenizer.encode("No", add_special_tokens=False)
        no_token_id = encoded[0] if encoded else None
        print(
            f"[INFO] Class weighting: 'No' token (id={no_token_id}) "
            f"upweighted ×{no_token_weight:.2f}"
        )

    # ── Resolve Yes/No token ids for compute_metrics (needed even if weight=1) ─
    if yes_token_id is None:
        encoded_yes = processor.tokenizer.encode("Yes", add_special_tokens=False)
        yes_token_id_for_metrics = encoded_yes[0] if encoded_yes else None
    else:
        yes_token_id_for_metrics = yes_token_id

    if no_token_id is None:
        encoded_no = processor.tokenizer.encode("No", add_special_tokens=False)
        no_token_id_for_metrics = encoded_no[0] if encoded_no else None
    else:
        no_token_id_for_metrics = no_token_id

    # ── Category-weighted sampler ─────────────────────────────────────────────
    cat_weights      = cfg.get("category_weights", {})
    weighted_sampler = None
    if cat_weights:
        import json as _json
        with open(data_json) as f:
            all_records = _json.load(f)
        import random as _random
        rng = _random.Random(42)
        shuffled = all_records[:]
        rng.shuffle(shuffled)
        n_val = max(1, int(len(shuffled) * d.get("val_fraction", 0.1)))
        train_records = shuffled[n_val:]
        weighted_sampler = make_weighted_sampler(train_records, cat_weights)
        print(f"[INFO] Category weights applied: {cat_weights}")

    # ── compute_metrics for F1-based checkpoint selection ────────────────────
    # Labels are stored as binary integers (1/0) by prediction_step, so
    # make_compute_metrics no longer needs token IDs — just the threshold.
    compute_metrics_fn = make_compute_metrics(threshold=0.3)
    print("[INFO] compute_metrics: F1 @ threshold=0.30 (binary labels via token decode)")

    # ── Training arguments ────────────────────────────────────────────────────
    use_bf16 = t.get("bf16", False) and torch.cuda.is_bf16_supported()
    use_fp16 = not use_bf16

    training_args = TrainingArguments(
        output_dir=str(project_dir / t["output_dir"]),
        num_train_epochs=t["num_train_epochs"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        per_device_eval_batch_size=t["per_device_eval_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        bf16=use_bf16,
        fp16=use_fp16,
        tf32=t["tf32"],
        eval_strategy=t["evaluation_strategy"],
        eval_steps=t.get("eval_steps", 100),
        save_strategy=t["save_strategy"],
        save_steps=t["save_steps"],
        save_total_limit=t["save_total_limit"],
        load_best_model_at_end=t.get("load_best_model_at_end", True),
        metric_for_best_model=t.get("metric_for_best_model", "eval_loss"),
        greater_is_better=t.get("greater_is_better", False),
        learning_rate=t["learning_rate"],
        weight_decay=t.get("weight_decay", 0.0),
        warmup_ratio=t["warmup_ratio"],
        lr_scheduler_type=t["lr_scheduler_type"],
        logging_steps=t["logging_steps"],
        gradient_checkpointing=t["gradient_checkpointing"],
        dataloader_num_workers=t["dataloader_num_workers"],
        report_to=t.get("report_to", "none"),
        remove_unused_columns=False,  # keep pixel_values
    )

    trainer = WeightedCETrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
        compute_metrics=compute_metrics_fn,
        # tokenizer is passed so prediction_step can call self.tokenizer.decode()
        # to convert true label token IDs to "y"/"n" strings — avoiding the
        # SentencePiece space-prefix ambiguity that made eval_f1 always 0.0.
        tokenizer=processor.tokenizer,
        yes_token_id=yes_token_id_for_metrics,
        yes_token_weight=yes_token_weight,
        # Always pass the resolved no_token_id (needed by prediction_step to
        # build the [yes_logit, no_logit] pair even when no_token_weight==1.0).
        # compute_loss only applies the weight when no_token_weight != 1.0.
        no_token_id=no_token_id_for_metrics,
        no_token_weight=no_token_weight,
        weighted_sampler=weighted_sampler,
        callbacks=[NormalizedLossCallback()],
    )

    # ── Resolve resume checkpoint ─────────────────────────────────────────────
    resume_from: str | bool = False
    if args.resume is True:
        output_path = project_dir / t["output_dir"]
        checkpoints = sorted(
            [d for d in output_path.glob("checkpoint-*") if d.is_dir()],
            key=lambda d: int(d.name.split("-")[-1]),
        )
        if checkpoints:
            resume_from = str(checkpoints[-1])
            print(f"[INFO] Auto-resuming from: {resume_from}")
        else:
            print("[WARN] --resume specified but no checkpoint-* dirs found in output_dir. Starting fresh.")
    elif args.resume:
        resume_from = str(Path(args.resume).resolve())
        print(f"[INFO] Resuming from: {resume_from}")

    print(f"[INFO] Config:         {args.config}")
    print(f"[INFO] Data:           {data_json}")
    print(f"[INFO] Output:         {t['output_dir']}")
    print(f"[INFO] report_to:      {t.get('report_to', 'none')}")
    print(f"[INFO] Yes-token wt:   ×{yes_token_weight:.2f}")
    print(f"[INFO] No-token wt:    ×{no_token_weight:.2f}")
    print(f"[INFO] Cat. weights:   {cat_weights or 'none'}")
    print(f"[INFO] Best metric:    {t.get('metric_for_best_model', 'eval_loss')} "
          f"(greater_is_better={t.get('greater_is_better', False)})")
    print(f"[INFO] Resume:         {resume_from or 'no (fresh start)'}")
    print("[INFO] Starting training...\n")

    trainer.train(resume_from_checkpoint=resume_from if resume_from else None)
    print("\n[INFO] Training complete!")


if __name__ == "__main__":
    main()
