#!/usr/bin/env python
"""Fail early if the English PG-19 server or model access is not ready."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_model_suite import select_models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--samples",
        type=Path,
        default=ROOT / "data/processed/samples.jsonl",
    )
    parser.add_argument("--models", default="english-pg19-a100")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-hub-check", action="store_true")
    return parser.parse_args()


def validate_samples(path: Path, limit: int | None) -> int:
    if not path.is_file():
        raise SystemExit(f"Sample file does not exist: {path}")
    ids: set[str] = set()
    words: Counter[str] = Counter()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            required = {"id", "word", "prefix", "rank", "source_path", "context_token_count"}
            missing = required - row.keys()
            if missing:
                raise SystemExit(
                    f"Sample row is missing {', '.join(sorted(missing))}: {path}"
                )
            sample_id = str(row["id"])
            if sample_id in ids:
                raise SystemExit(f"Duplicate sample id {sample_id!r}: {path}")
            ids.add(sample_id)
            words[str(row["word"])] += 1
            if limit is not None and len(ids) >= limit:
                break
    if not ids:
        raise SystemExit(f"No samples found in {path}")
    if limit is None and (len(ids) != 1000 or len(words) != 100 or set(words.values()) != {10}):
        raise SystemExit(
            "The full English PG-19 dataset must contain 1,000 samples: "
            "100 target words with 10 contexts each"
        )
    print(f"[preflight] dataset: {len(ids)} samples, {len(words)} target words")
    return len(ids)


def validate_runtime() -> None:
    import torch
    import transformers

    if transformers.__version__ != "5.12.1":
        raise SystemExit(
            f"Expected transformers 5.12.1, found {transformers.__version__}. "
            "Rerun the server launcher with INSTALL_DEPS=1."
        )
    if not torch.cuda.is_available():
        raise SystemExit(
            "PyTorch cannot see an NVIDIA GPU. Check nvidia-smi/driver visibility."
        )
    if not torch.cuda.is_bf16_supported():
        raise SystemExit(
            "The selected GPUs do not support bfloat16, which this paper run requires."
        )
    devices = [
        f"cuda:{index}={torch.cuda.get_device_name(index)}"
        for index in range(torch.cuda.device_count())
    ]
    print(f"[preflight] torch {torch.__version__}; {', '.join(devices)}")
    print(f"[preflight] transformers {transformers.__version__}")


def validate_hub_models(raw_models: str) -> None:
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoModelForMultimodalLM,
        AutoProcessor,
        AutoTokenizer,
    )

    failures: list[str] = []
    models = select_models(raw_models)
    print(f"[preflight] checking access/config/tokenizer for {len(models)} models")
    for index, model in enumerate(models, start=1):
        try:
            config = AutoConfig.from_pretrained(model.model_id)
            try:
                AutoModelForCausalLM._model_mapping[type(config)]
                loader = "causal-lm"
            except (KeyError, ValueError):
                AutoModelForMultimodalLM._model_mapping[type(config)]
                loader = "multimodal-lm"
            try:
                AutoTokenizer.from_pretrained(model.model_id)
            except (OSError, TypeError, ValueError):
                processor = AutoProcessor.from_pretrained(model.model_id)
                tokenizer = getattr(processor, "tokenizer", processor)
                if not hasattr(tokenizer, "encode") or not hasattr(tokenizer, "decode"):
                    raise TypeError("processor does not expose encode/decode")
            print(
                f"[preflight] {index:02d}/{len(models)} {model.slug}: "
                f"{config.model_type} via {loader}"
            )
        except Exception as error:  # noqa: BLE001 - collect every access/config failure.
            failures.append(f"{model.model_id}: {type(error).__name__}: {error}")

    if failures:
        formatted = "\n  - ".join(failures)
        raise SystemExit(
            "Model preflight failed. Accept any gated Llama/Gemma licenses and "
            f"confirm HF_TOKEN, then rerun:\n  - {formatted}"
        )


def main() -> None:
    args = parse_args()
    validate_samples(args.samples.resolve(), args.limit)
    validate_runtime()
    if not args.skip_hub_check:
        validate_hub_models(args.models)
    print("[preflight] all checks passed")


if __name__ == "__main__":
    main()
