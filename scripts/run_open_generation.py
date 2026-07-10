#!/usr/bin/env python
"""Resumable open-distribution generation and TTR/MTLD measurement."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path
from statistics import fmean, pstdev

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from run_model_suite import ModelSpec, select_models
from wcs.audit import load_hf_model_and_tokenizer, load_samples, parse_max_memory
from wcs.lexical_diversity import lexical_tokens, mtld, type_token_ratio


def select_contexts(samples: list[dict], limit: int, seed: int) -> list[dict]:
    if limit <= 0 or limit >= len(samples):
        return samples
    return random.Random(seed).sample(samples, limit)


def complete_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed = set()
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("complete"):
            completed.add(str(row["sample_id"]))
    return completed


def generate_model(model_spec: ModelSpec, samples: list[dict], args: argparse.Namespace) -> Path:
    import torch

    output = args.results_dir / f"generation.{model_spec.slug}.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    done = complete_ids(output) if args.resume else set()
    pending = [sample for sample in samples if str(sample["id"]) not in done]
    if not args.resume:
        output.write_text("", encoding="utf-8")
    if not pending:
        print(f"[skip] {model_spec.slug}: {len(done)} generations complete", flush=True)
        return output

    model, tokenizer = load_hf_model_and_tokenizer(
        model_spec.model_id,
        device=args.device,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        device_map=args.device_map,
        max_memory=parse_max_memory(args.max_memory),
        offload_folder=str(args.offload_folder) if args.offload_folder else None,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_device = args.device if not args.device_map else str(next(model.parameters()).device)
    max_new_tokens = args.max_new_tokens or math.ceil(args.words * 1.7) + 32
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    print(f"[run] {model_spec.slug}: {len(pending)} contexts, up to {max_new_tokens} tokens", flush=True)

    for start in range(0, len(pending), args.batch_size):
        batch = pending[start : start + args.batch_size]
        encoded = tokenizer(
            [sample["prefix"] for sample in batch],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_input_tokens,
        ).to(model_device)
        input_width = encoded["input_ids"].shape[1]
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                do_sample=True,
                temperature=args.temperature,
                top_k=0,
                top_p=1.0,
                min_new_tokens=max_new_tokens,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
            )
        with output.open("a", encoding="utf-8") as handle:
            for sample, sequence in zip(batch, generated):
                raw_text = tokenizer.decode(sequence[input_width:], skip_special_tokens=True)
                tokens = lexical_tokens(raw_text)[: args.words]
                text = " ".join(tokens)
                row = {
                    "sample_id": sample["id"],
                    "model": model_spec.model_id,
                    "model_slug": model_spec.slug,
                    "target_word": sample["word"],
                    "source_path": sample["source_path"],
                    "requested_words": args.words,
                    "word_count": len(tokens),
                    "temperature": args.temperature,
                    "top_k": 0,
                    "top_p": 1.0,
                    "ttr": type_token_ratio(tokens),
                    "mtld": mtld(tokens, threshold=args.mtld_threshold),
                    "text": text,
                    "complete": True,
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
        print(f"[progress] {model_spec.slug}: {min(start + len(batch), len(pending))}/{len(pending)}", flush=True)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return output


def summarize(paths: list[Path], output: Path, mtld_threshold: float) -> None:
    rows = []
    for path in paths:
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not records:
            continue
        all_tokens = [token for record in records for token in lexical_tokens(record["text"])]
        ttrs = [float(record["ttr"]) for record in records]
        mtlds = [float(record["mtld"]) for record in records]
        rows.append({
            "model": records[0]["model"],
            "model_slug": records[0]["model_slug"],
            "contexts": len(records),
            "total_words": len(all_tokens),
            "mean_ttr": fmean(ttrs),
            "sd_ttr": pstdev(ttrs),
            "mean_mtld": fmean(mtlds),
            "sd_mtld": pstdev(mtlds),
            "pooled_ttr": type_token_ratio(all_tokens),
            "pooled_mtld": mtld(all_tokens, threshold=mtld_threshold),
            "generation_path": str(path),
        })
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else ["model", "model_slug", "contexts"]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[summary] wrote {output}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--models", default=None)
    parser.add_argument("--contexts", type=int, default=50)
    parser.add_argument("--words", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--max-input-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--mtld-threshold", type=float, default=0.72)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--max-memory", default=None)
    parser.add_argument("--offload-folder", type=Path, default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.results_dir = args.results_dir.resolve()
    args.summary = args.summary or args.results_dir / "lexical_diversity.csv"
    samples = select_contexts(load_samples(args.samples), args.contexts, args.seed)
    models = select_models(args.models)
    paths = []
    for model_spec in models:
        paths.append(generate_model(model_spec, samples, args))
        summarize(paths, args.summary, args.mtld_threshold)
    summarize(paths, args.summary, args.mtld_threshold)


if __name__ == "__main__":
    main()
