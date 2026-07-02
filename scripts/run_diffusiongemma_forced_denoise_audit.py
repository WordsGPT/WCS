#!/usr/bin/env python
"""Run a forced-denoise WCS-style audit for DiffusionGemma.

DiffusionGemma is not an autoregressive next-token model. This runner therefore
does not implement the original WCS forced-path audit exactly. It measures the
rank/probability of each target token in the diffusion decoder canvas while the
prefix is encoded as context and previously audited target tokens are fixed in
earlier canvas positions.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from wcs.audit import (
    AuditTokenRow,
    decode_ranked_tokens,
    load_samples,
    rank_probability_from_logits,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a DiffusionGemma forced-denoise WCS-style audit.")
    parser.add_argument("--samples", type=Path, default=Path("data/processed/samples.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="google/diffusiongemma-26B-A4B-it")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260611)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument(
        "--denoising-step",
        type=int,
        default=None,
        help="Denoising step used for the official temperature schedule. Defaults to max_denoising_steps.",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def torch_dtype(name: str) -> Any:
    import torch

    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    return "auto"


def load_model_and_tokenizer(args: argparse.Namespace) -> tuple[Any, Any]:
    from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    model = DiffusionGemmaForBlockDiffusion.from_pretrained(
        args.model,
        dtype=torch_dtype(args.dtype),
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    )
    if not args.device_map:
        model.to(args.device)
    model.eval()
    return model, processor.tokenizer


def encode_prefix(tokenizer: Any, prefix: str) -> list[int]:
    token_ids = tokenizer.encode(prefix, add_special_tokens=False)
    if not token_ids:
        raise ValueError("Tokenizer produced no ids for sample prefix")
    return list(token_ids)


def encode_target_word(tokenizer: Any, word: str) -> list[int]:
    token_ids = tokenizer.encode(" " + word, add_special_tokens=False)
    if not token_ids:
        raise ValueError(f"Tokenizer produced no ids for target word {word!r}")
    return list(token_ids)


def diffusion_temperature(model: Any, args: argparse.Namespace) -> tuple[float, int, int]:
    if args.temperature is not None:
        max_steps = int(getattr(model.generation_config, "max_denoising_steps", 48) or 48)
        step = args.denoising_step if args.denoising_step is not None else max_steps
        return float(args.temperature), int(step), max_steps

    generation_config = model.generation_config
    max_steps = int(generation_config.max_denoising_steps)
    step = args.denoising_step if args.denoising_step is not None else max_steps
    if step <= 0 or step > max_steps:
        raise ValueError(f"--denoising-step must be in [1, {max_steps}]")
    t_min = float(generation_config.t_min)
    t_max = float(generation_config.t_max)
    return t_min + ((t_max - t_min) * (step / max_steps)), int(step), max_steps


def model_input_device(model: Any, fallback: str) -> Any:
    try:
        return model.device
    except AttributeError:
        return next(model.parameters()).device if fallback is None else fallback


def audit_sample(
    *,
    model: Any,
    tokenizer: Any,
    sample: dict[str, Any],
    model_name: str,
    device: Any,
    temperature: float,
    denoising_step: int,
    max_denoising_steps: int,
    generator: Any,
) -> list[dict[str, Any]]:
    import torch

    prefix_ids = encode_prefix(tokenizer, sample["prefix"])
    word_ids = encode_target_word(tokenizer, sample["word"])
    canvas_length = int(model.config.canvas_length)
    if len(word_ids) > canvas_length:
        raise ValueError(
            f"Target word {sample['word']!r} tokenizes to {len(word_ids)} tokens, "
            f"longer than DiffusionGemma canvas length {canvas_length}"
        )

    rows: list[dict[str, Any]] = []
    input_ids = torch.tensor([prefix_ids], dtype=torch.long, device=device)

    for token_index, token_id in enumerate(word_ids):
        canvas = torch.randint(
            low=0,
            high=int(model.config.text_config.vocab_size),
            size=(1, canvas_length),
            device=device,
            generator=generator,
        )
        if token_index:
            canvas[0, :token_index] = torch.tensor(word_ids[:token_index], dtype=torch.long, device=device)

        with torch.inference_mode():
            outputs = model(input_ids=input_ids, decoder_input_ids=canvas)
            logits = outputs.logits[0, token_index, :]

        result = rank_probability_from_logits(
            logits,
            int(token_id),
            temperature=temperature,
        )
        top_predictions = decode_ranked_tokens(tokenizer, result.top_tokens)
        row = AuditTokenRow(
            sample_id=sample["id"],
            model=model_name,
            word=sample["word"],
            word_rank=int(sample["rank"]),
            source_path=sample["source_path"],
            word_token_index=token_index,
            token_id=int(token_id),
            token_text=tokenizer.decode([token_id]),
            rank=result.rank,
            probability=result.probability,
            top_probability=result.top_probability,
            probability_ratio_to_top=result.probability_ratio_to_top,
            cumulative_probability=result.cumulative_probability,
            temperature=float(temperature),
            context_token_count=int(sample["context_token_count"]),
            prefix_char_count=len(sample["prefix"]),
            word_token_count=len(word_ids),
            top_5_tokens=[prediction.token_text for prediction in top_predictions],
            top_5_probs=[prediction.probability for prediction in top_predictions],
            rank_neighbors_above=decode_ranked_tokens(tokenizer, result.neighbors_above),
            rank_neighbors_below=decode_ranked_tokens(tokenizer, result.neighbors_below),
            rank_neighbor_count=5,
        )
        output = asdict(row)
        output.update(
            {
                "audit_protocol": "diffusiongemma_forced_denoise_v1",
                "diffusion_canvas_length": canvas_length,
                "diffusion_denoising_step": denoising_step,
                "diffusion_max_denoising_steps": max_denoising_steps,
                "diffusion_target_position": token_index,
                "diffusion_previous_target_tokens_fixed": token_index,
                "diffusion_current_target_token_noised": True,
                "diffusion_future_canvas_tokens_noised": True,
            }
        )
        rows.append(output)

    return rows


def iter_audit_rows(
    *,
    model: Any,
    tokenizer: Any,
    samples: Iterable[dict[str, Any]],
    model_name: str,
    device: Any,
    temperature: float,
    denoising_step: int,
    max_denoising_steps: int,
    seed: int,
) -> Iterable[dict[str, Any]]:
    import torch

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    for sample_index, sample in enumerate(samples, start=1):
        for row in audit_sample(
            model=model,
            tokenizer=tokenizer,
            sample=sample,
            model_name=model_name,
            device=device,
            temperature=temperature,
            denoising_step=denoising_step,
            max_denoising_steps=max_denoising_steps,
            generator=generator,
        ):
            row["diffusion_noise_seed"] = seed
            yield row
        print(f"[{sample_index}] {sample['id']} {sample['word']}", flush=True)


def main() -> None:
    args = parse_args()
    model, tokenizer = load_model_and_tokenizer(args)
    device = model_input_device(model, args.device)
    temperature, denoising_step, max_denoising_steps = diffusion_temperature(model, args)
    samples = load_samples(args.samples, limit=args.limit)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in iter_audit_rows(
            model=model,
            tokenizer=tokenizer,
            samples=samples,
            model_name=args.model,
            device=device,
            temperature=temperature,
            denoising_step=denoising_step,
            max_denoising_steps=max_denoising_steps,
            seed=args.seed,
        ):
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
    print(f"Wrote DiffusionGemma forced-denoise audit rows to {args.output}")


if __name__ == "__main__":
    main()
