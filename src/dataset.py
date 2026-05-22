"""CoNLL-2002 Spanish data loading, label alignment, and dataset preparation for token classification."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, MutableMapping

from datasets import DatasetDict, load_dataset
from transformers import PreTrainedTokenizerBase


def load_conll2002_spanish() -> DatasetDict:
    """Load the Spanish subset of CoNLL-2002 from the Hugging Face Hub."""
    # Hub dataset ships a loader script; datasets>=3 may require explicitly opting into remote code.
    return load_dataset("conll2002", "es", trust_remote_code=True)


def get_label_list() -> List[str]:
    """IOB2 label inventory used for Spanish NER (PII-oriented entity types)."""
    return [
        "O",
        "B-PER",
        "I-PER",
        "B-ORG",
        "I-ORG",
        "B-LOC",
        "I-LOC",
        "B-MISC",
        "I-MISC",
    ]


def tokenize_and_align_labels(
    examples: Mapping[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    label_all_tokens: bool = False,
    max_length: int | None = None,
) -> Dict[str, Any]:
    """
    Tokenize pre-split words and align word-level NER labels to subword tokens.

    Special tokens and (when ``label_all_tokens=False``) continuation subwords get label ``-100``
    so they are ignored in the loss (standard NER alignment).
    """
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        max_length=max_length,
        is_split_into_words=True,
    )

    all_labels: List[List[int]] = []
    for i, labels in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        aligned_labels: List[int] = []
        previous_word_idx: int | None = None
        for word_idx in word_ids:
            if word_idx is None:
                aligned_labels.append(-100)
            elif word_idx != previous_word_idx:
                aligned_labels.append(labels[word_idx])
            else:
                if label_all_tokens:
                    aligned_labels.append(labels[word_idx])
                else:
                    aligned_labels.append(-100)
            previous_word_idx = word_idx
        all_labels.append(aligned_labels)

    tokenized_inputs["labels"] = all_labels
    return tokenized_inputs


def _build_hf_label_maps(dataset: DatasetDict) -> tuple[Dict[int, int], List[str]]:
    """Map Hugging Face integer labels to our canonical label ids by string name."""
    label_feature = dataset["train"].features["ner_tags"].feature
    hf_names: List[str] = list(label_feature.names)
    canonical = get_label_list()
    try:
        hf_index_to_canonical_index = {
            hf_idx: canonical.index(name) for hf_idx, name in enumerate(hf_names)
        }
    except ValueError as exc:
        missing = [n for n in hf_names if n not in canonical]
        raise ValueError(
            f"Dataset labels {missing} are not covered by get_label_list(); "
            "update label definitions or dataset configuration."
        ) from exc
    return hf_index_to_canonical_index, hf_names


def _remap_ner_tags(batch: Mapping[str, Any], hf_to_canonical: Mapping[int, int]) -> Dict[str, Any]:
    return {
        **batch,
        "ner_tags": [[hf_to_canonical[int(tag)] for tag in seq] for seq in batch["ner_tags"]],
    }


def prepare_datasets(
    tokenizer: PreTrainedTokenizerBase,
    config: MutableMapping[str, Any],
) -> DatasetDict:
    """
    Return train/validation/test splits tokenized and ready for ``Trainer``.

    Config keys used: ``dataset.max_length`` (passed through tokenizer padding/truncation via
    dataset map batching), ``dataset`` subset metadata for loading only.
    """
    dataset_cfg = config["dataset"]
    raw = load_conll2002_spanish()
    hf_to_canonical, _ = _build_hf_label_maps(raw)

    splits = DatasetDict(
        {
            "train": raw["train"],
            "validation": raw["validation"],
            "test": raw["test"],
        }
    )

    splits = splits.map(
        lambda batch: _remap_ner_tags(batch, hf_to_canonical),
        batched=True,
    )

    max_length = int(dataset_cfg["max_length"])

    def _tokenize_batch(batch: Mapping[str, Any]) -> Dict[str, Any]:
        return tokenize_and_align_labels(
            batch,
            tokenizer,
            label_all_tokens=False,
            max_length=max_length,
        )

    tokenized = splits.map(
        _tokenize_batch,
        batched=True,
        remove_columns=splits["train"].column_names,
        desc="Tokenizing and aligning labels",
    )

    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    return tokenized
