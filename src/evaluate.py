"""Evaluate a saved LoRA adapter on CoNLL-2002 Spanish without quantization."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import torch
import yaml
from peft import PeftModel
from seqeval.metrics import classification_report
from transformers import AutoModelForTokenClassification

from dataset import get_label_list, prepare_datasets
from model import load_tokenizer
from ner_eval import evaluate_ner_split


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_config(config_path: Path) -> Dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        data: Dict[str, Any] = yaml.safe_load(handle)
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved QLoRA adapter for Spanish NER.")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to saved adapter weights (e.g., ./outputs/best_model).",
    )
    args = parser.parse_args()

    config_path = _project_root() / "config.yaml"
    config = load_config(config_path)
    model_cfg = config["model"]
    training_cfg = config["training"]

    labels = get_label_list()
    num_labels = int(model_cfg["num_labels"])
    if len(labels) != num_labels:
        raise ValueError("Label list length does not match config model.num_labels.")

    label2id: Dict[str, int] = {label: idx for idx, label in enumerate(labels)}
    id2label: Dict[int, str] = {idx: label for label, idx in label2id.items()}

    tokenizer = load_tokenizer(config)
    datasets = prepare_datasets(tokenizer, config)

    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    base_model = AutoModelForTokenClassification.from_pretrained(
        model_cfg["base_model"],
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        torch_dtype=torch_dtype,
    )

    model = PeftModel.from_pretrained(base_model, args.model_path)
    model.eval()

    artifacts = evaluate_ner_split(
        model,
        tokenizer,
        datasets["test"],
        id2label=id2label,
        metric_output_dir=str(training_cfg["output_dir"]),
        batch_size=int(training_cfg["per_device_eval_batch_size"]),
        fp16=torch.cuda.is_available(),
        use_cpu_eval=not torch.cuda.is_available(),
    )

    report = classification_report(artifacts.true_sequences, artifacts.pred_sequences)
    print(report)
    print(f"Overall precision: {artifacts.precision:.6f}")
    print(f"Overall recall: {artifacts.recall:.6f}")
    print(f"Overall F1: {artifacts.f1:.6f}")


if __name__ == "__main__":
    main()
