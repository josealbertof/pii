from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping

import numpy as np
import torch
from seqeval.metrics import f1_score, precision_score, recall_score
from torch import nn
from torch.utils.data import DataLoader
from transformers import (
    DataCollatorForTokenClassification,
    EvalPrediction,
    PreTrainedTokenizerBase,
)


@dataclass(frozen=True)
class NEREvalArtifacts:
    precision: float
    recall: float
    f1: float
    true_sequences: List[List[str]]
    pred_sequences: List[List[str]]


def _as_logits(predictions: Any) -> np.ndarray:
    logits = predictions[0] if isinstance(predictions, tuple) else predictions
    return np.asarray(logits)


def compute_metrics_builder(id2label: Mapping[int, str]):
    """Same metric builder as trainer training (seqeval entity F1 over word-aligned tags)."""

    def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
        logits = _as_logits(eval_pred.predictions)
        labels = eval_pred.label_ids
        predictions = np.argmax(logits, axis=-1)

        true_sequences: List[List[str]] = []
        pred_sequences: List[List[str]] = []

        for pred_row, label_row in zip(predictions, labels):
            true_tags: List[str] = []
            pred_tags: List[str] = []
            for pred_id, lab_id in zip(pred_row, label_row):
                if lab_id == -100:
                    continue
                true_tags.append(id2label[int(lab_id)])
                pred_tags.append(id2label[int(pred_id)])
            true_sequences.append(true_tags)
            pred_sequences.append(pred_tags)

        return {
            "precision": float(precision_score(true_sequences, pred_sequences)),
            "recall": float(recall_score(true_sequences, pred_sequences)),
            "f1": float(f1_score(true_sequences, pred_sequences)),
        }

    return compute_metrics


def _eval_device(model: nn.Module, force_cpu: bool) -> torch.device:
    if force_cpu:
        return torch.device("cpu")
    device = getattr(model, "device", None)
    if isinstance(device, torch.device):
        return device
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def evaluate_ner_split(
    model: nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    eval_dataset,
    *,
    id2label: Mapping[int, str],
    metric_output_dir: str,
    batch_size: int,
    fp16: bool | None,
    use_cpu_eval: bool = False,
) -> NEREvalArtifacts:
    """
    Token-level inference + seqeval entity metrics.

    We avoid ``Trainer.predict`` here: ``Trainer`` raises on bitsandbytes-quantized **base** models
    (no PEFT adapter) even for evaluation-only runs, because training on “purely quantized” weights
    is unsupported. A manual forward loop is equivalent for metrics.
    """

    Path(metric_output_dir).mkdir(parents=True, exist_ok=True)

    if use_cpu_eval:
        effective_fp16 = False
    else:
        effective_fp16 = fp16 if fp16 is not None else torch.cuda.is_available()

    is_bnb_quant = getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False)
    use_amp = bool(effective_fp16 and torch.cuda.is_available() and not use_cpu_eval and not is_bnb_quant)

    collator = DataCollatorForTokenClassification(tokenizer)
    loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )

    device = _eval_device(model, use_cpu_eval)
    model.eval()

    # Per-example preds: batches are padded to each batch max length only, so raw batch logits have
    # different seq dims across batches and cannot be concatenated naively along the batch axis.
    pred_rows: List[np.ndarray] = []
    label_rows: List[np.ndarray] = []

    for batch in loader:
        labels = batch["labels"]
        inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}

        with torch.inference_mode():
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = model(**inputs)
            else:
                outputs = model(**inputs)

        logits_b = outputs.logits.detach().float().cpu().numpy()
        attn = inputs["attention_mask"].detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()

        batch_size = int(labels_np.shape[0])
        for i in range(batch_size):
            length = int(attn[i].sum())
            pred_rows.append(np.argmax(logits_b[i, :length], axis=-1))
            label_rows.append(labels_np[i, :length])

    true_sequences: List[List[str]] = []
    pred_sequences: List[List[str]] = []

    for pred_row, label_row in zip(pred_rows, label_rows):
        true_tags: List[str] = []
        pred_tags: List[str] = []
        for pred_id, lab_id in zip(pred_row, label_row):
            if lab_id == -100:
                continue
            true_tags.append(id2label[int(lab_id)])
            pred_tags.append(id2label[int(pred_id)])
        true_sequences.append(true_tags)
        pred_sequences.append(pred_tags)

    return NEREvalArtifacts(
        precision=float(precision_score(true_sequences, pred_sequences)),
        recall=float(recall_score(true_sequences, pred_sequences)),
        f1=float(f1_score(true_sequences, pred_sequences)),
        true_sequences=true_sequences,
        pred_sequences=pred_sequences,
    )
