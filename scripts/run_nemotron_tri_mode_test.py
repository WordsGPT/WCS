#!/usr/bin/env python
"""Smoke/benchmark runner for Nemotron-Labs-Diffusion tri-mode decoding.

This tests the model's execution modes, not WCS top-k/top-p/min-p reachability.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_MODEL = "nvidia/Nemotron-Labs-Diffusion-14B"
DEFAULT_PROMPT = "Write a concise paragraph explaining why lexical diversity matters in language models."
MODES = ("ar", "diffusion", "self_spec")


@dataclass(frozen=True)
class ModeResult:
    model: str
    mode: str
    prompt: str
    output: str
    prompt_tokens: int
    output_tokens: int
    elapsed_seconds: float
    tokens_per_second: float
    nfe: int | float | None
    max_new_tokens: int
    block_length: int
    threshold: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run AR, diffusion, and self-speculation tests for Nemotron-Labs-Diffusion."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--mode", choices=(*MODES, "all"), default="all")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSONL result path.")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--block-length", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("auto", "bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--device-map", default=None, help="Optional Transformers device_map, e.g. auto.")
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--linear-spec-lora",
        action="store_true",
        help="Attach the model card's linear_spec_lora adapter for self-speculation.",
    )
    return parser.parse_args()


def torch_dtype(name: str) -> Any:
    import torch

    return {
        "auto": "auto",
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def load_model_and_tokenizer(args: argparse.Namespace) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    dtype = torch_dtype(args.dtype)
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": args.trust_remote_code,
        "dtype": dtype,
    }
    if args.device_map:
        model_kwargs["device_map"] = args.device_map
        model_kwargs["low_cpu_mem_usage"] = True

    try:
        model = AutoModel.from_pretrained(args.model, **model_kwargs)
    except TypeError as error:
        if "dtype" not in str(error):
            raise
        model_kwargs["torch_dtype"] = model_kwargs.pop("dtype")
        model = AutoModel.from_pretrained(args.model, **model_kwargs)
    if args.linear_spec_lora:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise RuntimeError("--linear-spec-lora requires `pip install peft`") from exc
        model = PeftModel.from_pretrained(model, args.model, subfolder="linear_spec_lora").eval().model

    if not args.device_map:
        model = model.to(args.device)
    model.eval()
    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return model, tokenizer


def build_prompt_ids(tokenizer: Any, prompt: str, device: str) -> Any:
    import torch

    history = [{"role": "user", "content": prompt}]
    rendered = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
    ids = tokenizer(rendered, return_tensors="pt").input_ids
    if device.startswith("cuda") and torch.cuda.is_available():
        ids = ids.to(device="cuda")
    else:
        ids = ids.to(device=device)
    return ids


def call_mode(model: Any, prompt_ids: Any, tokenizer: Any, args: argparse.Namespace, mode: str) -> tuple[Any, Any]:
    common = {
        "max_new_tokens": args.max_new_tokens,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if mode == "ar":
        return model.ar_generate(prompt_ids, **common)
    if mode == "diffusion":
        return model.generate(
            prompt_ids,
            block_length=args.block_length,
            threshold=args.threshold,
            **common,
        )
    if mode == "self_spec":
        return model.linear_spec_generate(
            prompt_ids,
            block_length=args.block_length,
            **common,
        )
    raise ValueError(f"Unsupported mode: {mode}")


def synchronize_if_needed(device: str) -> None:
    import torch

    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def run_mode(model: Any, tokenizer: Any, args: argparse.Namespace, mode: str) -> ModeResult:
    prompt_ids = build_prompt_ids(tokenizer, args.prompt, args.device)
    synchronize_if_needed(args.device)
    started = time.perf_counter()
    with __import__("torch").inference_mode():
        out_ids, nfe = call_mode(model, prompt_ids, tokenizer, args, mode)
    synchronize_if_needed(args.device)
    elapsed = time.perf_counter() - started

    generated_ids = out_ids[:, prompt_ids.shape[1] :]
    text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    output_tokens = int(generated_ids.shape[1])
    return ModeResult(
        model=args.model,
        mode=mode,
        prompt=args.prompt,
        output=text,
        prompt_tokens=int(prompt_ids.shape[1]),
        output_tokens=output_tokens,
        elapsed_seconds=elapsed,
        tokens_per_second=(output_tokens / elapsed) if elapsed > 0 else 0.0,
        nfe=nfe.item() if hasattr(nfe, "item") else nfe,
        max_new_tokens=args.max_new_tokens,
        block_length=args.block_length,
        threshold=args.threshold,
    )


def write_results(path: Path, results: list[ModeResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")


def main() -> int:
    args = parse_args()
    selected_modes = list(MODES) if args.mode == "all" else [args.mode]
    model, tokenizer = load_model_and_tokenizer(args)

    results = []
    for mode in selected_modes:
        result = run_mode(model, tokenizer, args, mode)
        results.append(result)
        print(
            f"[{mode}] {result.output_tokens} tokens in {result.elapsed_seconds:.2f}s "
            f"({result.tokens_per_second:.2f} tok/s, NFE={result.nfe})"
        )
        print(result.output.strip())
        print()

    if args.output:
        write_results(args.output, results)
        print(f"Wrote JSONL results to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
