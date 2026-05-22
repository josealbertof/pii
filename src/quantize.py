from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List

import torch
import yaml
from transformers import AutoModelForTokenClassification, AutoTokenizer, BitsAndBytesConfig

from dataset import get_label_list, prepare_datasets
from ner_eval import evaluate_ner_split

logger = logging.getLogger(__name__)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_config(config_path: Path) -> Dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        data: Dict[str, Any] = yaml.safe_load(handle)
    return data


def _bnb_compute_dtype(name: str) -> torch.dtype:
    dtype = getattr(torch, name, None)
    if dtype is None or not isinstance(dtype, torch.dtype):
        raise ValueError(f"Unsupported torch dtype for quantization compute: {name!r}")
    return dtype


def _quantize_dynamic_torch(model_cpu: AutoModelForTokenClassification) -> Any:
    """
    Torch dynamic weight-only quantization for ``nn.Linear``.
    Inference is evaluated on CPU to match quantization runtime expectations.
    """
    if hasattr(torch, "ao") and hasattr(torch.ao, "quantization"):
        backend = torch.ao.quantization
    else:
        try:
            from torch import quantization as torch_quant
        except ImportError:
            torch_quant = None

        backend = torch_quant

    if backend is None or not hasattr(backend, "quantize_dynamic"):
        raise RuntimeError("Torch dynamic quantization helpers are unavailable in this build.")

    model_cpu.eval()
    return backend.quantize_dynamic(
        model_cpu,
        {torch.nn.Linear},
        dtype=torch.qint8,
    )


def _try_run(
    name: str,
    grain: str,
    device_hint: str,
    factory: Callable[[], Any],
    tokenizer,
    test_dataset,
    id2label: Dict[int, str],
    *,
    staging_dir: str,
    batch_size: int,
    fp16_inference: bool | None,
    skip_when_cuda_needed_but_missing: bool,
    use_cpu_eval: bool,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "name": name,
        "granularity": grain,
        "device_hint": device_hint,
        "precision": "",
        "recall": "",
        "f1": "",
        "error": "",
    }

    cuda_missing_but_required = skip_when_cuda_needed_but_missing and not torch.cuda.is_available()
    cuda_expected = device_hint == "cuda"
    if cuda_expected and cuda_missing_but_required:
        payload["error"] = "skipped: CUDA unavailable"
        return payload

    try:
        model = factory()
        model.eval()

        staging = str(Path(staging_dir) / f"quantize_tmp_{name.replace(' ', '_')}")
        result = evaluate_ner_split(
            model,
            tokenizer,
            test_dataset,
            id2label=id2label,
            metric_output_dir=staging,
            batch_size=batch_size,
            fp16=fp16_inference,
            use_cpu_eval=use_cpu_eval,
        )
        payload["precision"] = f"{result.precision:.6f}"
        payload["recall"] = f"{result.recall:.6f}"
        payload["f1"] = f"{result.f1:.6f}"

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except Exception as exc:
        payload["error"] = str(exc)
        logger.warning("Variant %s failed: %s", name, exc, exc_info=logger.isEnabledFor(logging.DEBUG))

    return payload


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        description="Compare test F1 of a merged NER model versus multiple quantized loaders.",
    )
    parser.add_argument(
        "--merged_model",
        type=str,
        default=None,
        help="Path saved by train.py merge step (default: <training.output_dir>/merged_fp16_model).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional path to YAML config (default: repo config.yaml).",
    )
    args = parser.parse_args()

    project_root = _project_root()
    config_path = Path(args.config) if args.config else project_root / "config.yaml"
    config = load_config(config_path)
    training_cfg = config["training"]
    model_cfg = config["model"]
    quant_yaml = config["quantization"]

    merged_default = Path(training_cfg["output_dir"]).resolve() / "merged_fp16_model"
    merged_path = Path(args.merged_model).resolve() if args.merged_model else merged_default
    if not merged_path.is_dir():
        raise FileNotFoundError(
            f"Merged model folder not found: {merged_path}. "
            "Run train.py first; it saves merged FP16 weights for post-training quantization benchmarks.",
        )

    labels = get_label_list()
    if len(labels) != int(model_cfg["num_labels"]):
        raise ValueError("Label inventory length mismatches config model.num_labels.")

    id2label: Dict[int, str] = {idx: lab for idx, lab in enumerate(labels)}

    tokenizer = AutoTokenizer.from_pretrained(str(merged_path), use_fast=True, add_prefix_space=True)
    datasets = prepare_datasets(tokenizer, config)

    batch_size_cpu = max(8, min(int(training_cfg["per_device_eval_batch_size"]), 64))
    batch_size_gpu = int(training_cfg["per_device_eval_batch_size"])
    staging_root = Path(training_cfg["output_dir"]).resolve() / "quantize_benchmark"
    staging_root.mkdir(parents=True, exist_ok=True)

    @dataclass(frozen=True)
    class BenchCase:
        name: str
        grain: str
        device_hint: str
        factory: Callable[[], Any]
        batch_size: int
        fp16: bool | None
        skip_when_cuda_missing: bool
        use_cpu_eval: bool

    cases: List[BenchCase] = []
    cuda_ok = torch.cuda.is_available()

    cases.append(
        BenchCase(
            name="baseline_fp16_merged",
            grain="FP16/FP32 full weights (LoRA merged; same dtype as training export)",
            device_hint="cuda" if cuda_ok else "cpu",
            factory=lambda: AutoModelForTokenClassification.from_pretrained(
                str(merged_path),
                torch_dtype=torch.float16 if cuda_ok else torch.float32,
                device_map="auto" if cuda_ok else {"": torch.device("cpu")},
            ),
            batch_size=batch_size_gpu if cuda_ok else batch_size_cpu,
            fp16=cuda_ok,
            skip_when_cuda_missing=False,
            use_cpu_eval=not cuda_ok,
        )
    )

    if cuda_ok:
        compute_dtype_name = str(quant_yaml["bnb_4bit_compute_dtype"])
        bnb_compute = _bnb_compute_dtype(compute_dtype_name)

        def _bnb_factory(cfg: BitsAndBytesConfig) -> Callable[[], Any]:
            return lambda: AutoModelForTokenClassification.from_pretrained(
                str(merged_path),
                quantization_config=cfg,
                device_map="auto",
            )

        cases.extend(
            [
                BenchCase(
                    name="bitsandbytes_8bit",
                    grain="8-bit weights (bitsandbytes load_in_8bit)",
                    device_hint="cuda",
                    factory=_bnb_factory(BitsAndBytesConfig(load_in_8bit=True)),
                    batch_size=batch_size_gpu,
                    fp16=False,
                    skip_when_cuda_missing=True,
                    use_cpu_eval=False,
                ),
                BenchCase(
                    name="bitsandbytes_4bit_nf4",
                    grain="4-bit NF4 weights (bnb; double_quant from config when applicable)",
                    device_hint="cuda",
                    factory=_bnb_factory(
                        BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_compute_dtype=bnb_compute,
                            bnb_4bit_quant_type="nf4",
                            bnb_4bit_use_double_quant=bool(quant_yaml["bnb_4bit_use_double_quant"]),
                        )
                    ),
                    batch_size=batch_size_gpu,
                    fp16=False,
                    skip_when_cuda_missing=True,
                    use_cpu_eval=False,
                ),
                BenchCase(
                    name="bitsandbytes_4bit_fp4",
                    grain="4-bit FP4 weights (bnb fp4 stack; GPU-dependent)",
                    device_hint="cuda",
                    factory=_bnb_factory(
                        BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_compute_dtype=bnb_compute,
                            bnb_4bit_quant_type="fp4",
                            bnb_4bit_use_double_quant=bool(quant_yaml["bnb_4bit_use_double_quant"]),
                        )
                    ),
                    batch_size=batch_size_gpu,
                    fp16=False,
                    skip_when_cuda_missing=True,
                    use_cpu_eval=False,
                ),
            ]
        )

    def _torch_dynamic_factory() -> Any:
        model_cpu = AutoModelForTokenClassification.from_pretrained(
            str(merged_path),
            torch_dtype=torch.float32,
            device_map={"": torch.device("cpu")},
        )
        return _quantize_dynamic_torch(model_cpu)

    cases.append(
        BenchCase(
            name="torch_dynamic_linear_int8_cpu",
            grain="PyTorch dynamic qint8 on Linear (CPU eval; weight-only)",
            device_hint="cpu",
            factory=_torch_dynamic_factory,
            batch_size=batch_size_cpu,
            fp16=False,
            skip_when_cuda_missing=False,
            use_cpu_eval=True,
        )
    )

    rows: List[Dict[str, Any]] = []
    for case in cases:
        rows.append(
            _try_run(
                case.name,
                case.grain,
                case.device_hint,
                case.factory,
                tokenizer,
                datasets["test"],
                id2label,
                staging_dir=str(staging_root),
                batch_size=case.batch_size,
                fp16_inference=case.fp16,
                skip_when_cuda_needed_but_missing=case.skip_when_cuda_missing,
                use_cpu_eval=case.use_cpu_eval,
            )
        )

    width_name = 34
    width_grain = 52
    print()
    print("CoNLL-2002 Spanish `test` — merged checkpoint vs quantized inference paths")
    print(f"checkpoint: {merged_path}")
    print()
    header = (
        f"{'method':<{width_name}} {'granularity':<{width_grain}} "
        f"{'dev':<5} {'P':>10} {'R':>10} {'F1':>10}  notes"
    )
    print(header)
    print("-" * len(header))

    for row in rows:
        notes = row["error"] if row["error"] else ""
        print(
            f"{row['name']:<{width_name}} {row['granularity']:<{width_grain}} "
            f"{row['device_hint']:<5} {str(row['precision']):>10} {str(row['recall']):>10} "
            f"{str(row['f1']):>10}  {notes}"
        )
    print()


if __name__ == "__main__":
    main()
