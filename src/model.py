from __future__ import annotations

import logging
from typing import Any, Dict, Mapping, Tuple

import torch
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from dataset import get_label_list

logger = logging.getLogger(__name__)


def _bnb_compute_dtype(name: str) -> torch.dtype:
    dtype = getattr(torch, name, None)
    if dtype is None or not isinstance(dtype, torch.dtype):
        raise ValueError(f"Unsupported torch dtype for quantization: {name!r}")
    return dtype


def _task_type_from_config(value: str) -> TaskType:
    try:
        return TaskType[value]
    except KeyError as exc:
        raise ValueError(f"Unsupported PEFT TaskType: {value!r}") from exc


def load_tokenizer(config: Mapping[str, Any]) -> PreTrainedTokenizerBase:
    """Load the tokenizer with RoBERTa-compatible whitespace handling."""
    model_cfg = config["model"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["base_model"],
        use_fast=True,
        add_prefix_space=True,
    )
    return tokenizer


def load_quantized_model(config: Mapping[str, Any]) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load a 4-bit quantized token classification model and attach LoRA adapters."""
    model_cfg = config["model"]
    qlora_cfg = config["qlora"]
    quant_cfg = config["quantization"]

    tokenizer = load_tokenizer(config)

    label_list = get_label_list()
    expected = int(model_cfg["num_labels"])
    if len(label_list) != expected:
        raise ValueError(
            f"model.num_labels ({expected}) must match len(get_label_list()) ({len(label_list)})."
        )

    label2id: Dict[str, int] = {label: idx for idx, label in enumerate(label_list)}
    id2label: Dict[int, str] = {idx: label for label, idx in label2id.items()}

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=bool(quant_cfg["load_in_4bit"]),
        bnb_4bit_compute_dtype=_bnb_compute_dtype(str(quant_cfg["bnb_4bit_compute_dtype"])),
        bnb_4bit_quant_type=str(quant_cfg["bnb_4bit_quant_type"]),
        bnb_4bit_use_double_quant=bool(quant_cfg["bnb_4bit_use_double_quant"]),
    )

    model = AutoModelForTokenClassification.from_pretrained(
        model_cfg["base_model"],
        num_labels=expected,
        id2label=id2label,
        label2id=label2id,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=int(qlora_cfg["r"]),
        lora_alpha=int(qlora_cfg["lora_alpha"]),
        lora_dropout=float(qlora_cfg["lora_dropout"]),
        bias=str(qlora_cfg["bias"]),
        target_modules=list(qlora_cfg["target_modules"]),
        task_type=_task_type_from_config(str(qlora_cfg["task_type"])),
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, tokenizer
