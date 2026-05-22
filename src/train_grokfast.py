from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any, Dict

import torch
import yaml
from peft import PeftModel
from transformers import (
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

from dataset import get_label_list, prepare_datasets
from model import load_quantized_model
from ner_eval import compute_metrics_builder
from train.grokfast_trainer import GrokfastTrainer

logger = logging.getLogger(__name__)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_config(config_path: Path) -> Dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        data: Dict[str, Any] = yaml.safe_load(handle)
    return data


def _parse_cli_args(default_config_path: Path) -> argparse.Namespace:
    training_defaults = load_config(default_config_path).get("training", {})
    parser = argparse.ArgumentParser(
        description="Fine-tune Spanish NER (QLoRA). Por defecto se leen épocas, lr y fp16 desde config YAML.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=default_config_path,
        help="Ruta al config.yaml del proyecto (predeterminado: config.yaml en la raíz del repo).",
    )
    parser.add_argument(
        "--epochs",
        type=float,
        default=None,
        metavar="N",
        help=f"Épocas de entreno (TrainingArguments.num_train_epochs). Por defecto las definidas en config.yaml.",
    )
    parser.add_argument(
        "--learning-rate",
        "--lr",
        type=float,
        default=None,
        dest="learning_rate",
        metavar="RATE",
        help=f"Tasa de aprendizaje.",
    )
    parser.add_argument(
        "--fp16",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Usar FP16 en el Trainer cuando hay soporte (--fp16=True, --no-fp16=False). "
            "Si omites esta opción se usa el campo training.fp16 del config YAML "
            f"(actualmente en el predeterminado: {training_defaults['fp16']})."
        ),
    )

    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    default_config = _project_root() / "config.yaml"
    cli = _parse_cli_args(default_config)
    config_path = cli.config
    config = load_config(config_path)

    training_cfg = config["training"]
    model_cfg = config["model"]
    output_dir = str(training_cfg["output_dir"])

    num_train_epochs = float(cli.epochs if cli.epochs is not None else training_cfg["num_train_epochs"])
    learning_rate = float(cli.learning_rate if cli.learning_rate is not None else training_cfg["learning_rate"])
    use_fp16 = bool(training_cfg["fp16"]) if cli.fp16 is None else bool(cli.fp16)

    logger.info(
        "Trainer overrides: epochs=%s, learning_rate=%s, fp16=%s (config: %s)",
        num_train_epochs,
        learning_rate,
        use_fp16,
        config_path,
    )

    model, tokenizer = load_quantized_model(config)
    datasets = prepare_datasets(tokenizer, config)

    labels = get_label_list()
    id2label = {i: lab for i, lab in enumerate(labels)}
    compute_metrics = compute_metrics_builder(id2label)

    label2id: Dict[str, int] = {label: idx for idx, label in enumerate(labels)}

    training_kwargs: Dict[str, Any] = {
        "output_dir": output_dir,
        "num_train_epochs": num_train_epochs,
        "per_device_train_batch_size": int(training_cfg["per_device_train_batch_size"]),
        "per_device_eval_batch_size": int(training_cfg["per_device_eval_batch_size"]),
        "learning_rate": learning_rate,
        "weight_decay": float(training_cfg["weight_decay"]),
        "warmup_ratio": float(training_cfg["warmup_ratio"]),
        "lr_scheduler_type": str(training_cfg["lr_scheduler_type"]),
        "logging_steps": int(training_cfg["logging_steps"]),
        "save_strategy": str(training_cfg["save_strategy"]),
        "eval_strategy": str(training_cfg["evaluation_strategy"]),
        "load_best_model_at_end": bool(training_cfg["load_best_model_at_end"]),
        "metric_for_best_model": str(training_cfg["metric_for_best_model"]),
        "greater_is_better": True,
        "fp16": use_fp16,
        "seed": int(training_cfg["seed"]),
        "report_to": [],
    }

    args = TrainingArguments(**training_kwargs)

    data_collator = DataCollatorForTokenClassification(tokenizer)

    trainer = GrokfastTrainer(
        model=model,
        args=args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        # Grokfast-specific args:
        grokfast_type="ma",    # "ema" or "ma"
        grokfast_alpha=0.1,    # EMA momentum (ignored if type="ma")
        grokfast_lamb=0.5,      # Amplification factor
        grokfast_window_size=100,  # Only relevant for type="ma"
    )
    trainer.train()

    best_dir = os.path.join(output_dir, "best_model")
    os.makedirs(best_dir, exist_ok=True)
    trainer.save_model(best_dir)
    tokenizer.save_pretrained(best_dir)
    logger.info("Saved trainer checkpoint under %s.", best_dir)

    merged_dir = os.path.join(output_dir, "merged_fp16_model")
    os.makedirs(merged_dir, exist_ok=True)
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    num_labels = int(model_cfg["num_labels"])
    logger.info(
        "Merging LoRA into full-precision weights for quantization/eval artifacts → %s",
        merged_dir,
    )
    base_for_merge = AutoModelForTokenClassification.from_pretrained(
        model_cfg["base_model"],
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        torch_dtype=torch_dtype,
    )
    peft_bundle = PeftModel.from_pretrained(base_for_merge, best_dir)
    merged_model = peft_bundle.merge_and_unload()
    merged_model.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)
    logger.info("Merged model saved under %s (use scripts/quantize or evaluate from here).", merged_dir)

    best_f1 = getattr(trainer.state, "best_metric", None)
    logger.info("Training complete.")
    if best_f1 is not None:
        print(f"Best validation F1 ({training_cfg['metric_for_best_model']}): {best_f1:.6f}")
    else:
        print("Best validation F1 could not be determined from trainer.state.best_metric.")


if __name__ == "__main__":
    main()
