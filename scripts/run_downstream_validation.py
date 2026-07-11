#!/usr/bin/env python
"""Run matched WCS and lexical-diversity validation on FineWeb contexts.

The experiment varies one decoding filter at a time, uses the same contexts
and prompt representation for forced-path WCS and open generation, and writes
configuration-level diversity summaries plus WCS/diversity correlations.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean, median, pstdev

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from run_model_suite import ModelSpec, select_models
from wcs.audit import (
    audit_sample_temperatures,
    load_hf_model_and_tokenizer,
    load_samples,
    parse_max_memory,
    parse_temperature_list,
)
from wcs.lexical_diversity import lexical_tokens, mtld, type_token_ratio
from wcs.metrics import summarize_wcs, write_summary_csv


DEFAULT_TOP_K = (10, 20, 50, 80)
DEFAULT_TOP_P = (0.80, 0.90, 0.95, 0.99)
DEFAULT_MIN_P = (0.01, 0.05, 0.10)


@dataclass(frozen=True)
class Condition:
    decoder: str
    parameter: float

    @property
    def key(self) -> str:
        return f"{self.decoder}:{self.parameter:g}"

    def generation_kwargs(self) -> dict[str, float | int]:
        if self.decoder == "top_k":
            return {"top_k": int(self.parameter), "top_p": 1.0, "min_p": 0.0}
        if self.decoder == "top_p":
            return {"top_k": 0, "top_p": float(self.parameter), "min_p": 0.0}
        if self.decoder == "min_p":
            return {"top_k": 0, "top_p": 1.0, "min_p": float(self.parameter)}
        if self.decoder == "untruncated":
            return {"top_k": 0, "top_p": 1.0, "min_p": 0.0}
        raise ValueError(f"Unknown decoder {self.decoder!r}")


def parse_values(raw: str, kind: type = float) -> list:
    values = [kind(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("At least one value is required")
    return values


def conditions_from_args(args: argparse.Namespace) -> list[Condition]:
    values = [
        *(Condition("top_k", float(value)) for value in parse_values(args.top_k_values, int)),
        *(Condition("top_p", value) for value in parse_values(args.top_p_values)),
        *(Condition("min_p", value) for value in parse_values(args.min_p_values)),
    ]
    if args.include_untruncated:
        values.append(Condition("untruncated", 1.0))
    return values


def select_contexts(samples: list[dict], limit: int, seed: int) -> list[dict]:
    if limit <= 0 or limit >= len(samples):
        return samples
    return random.Random(seed).sample(samples, limit)


def continuation_instruction(prefix: str, requested_words: int) -> str:
    return (
        f"Continue the passage below naturally for at least {requested_words + 50} words. "
        "Preserve its language, topic, style, and coherence. Output only the continuation.\n\n"
        f"PASSAGE:\n{prefix}"
    )


def _is_chat_model(model_spec: ModelSpec) -> bool:
    """Return True for instruct/distill variants that need chat templates."""
    return model_spec.variant in ("instruct", "distill")


def _raw_prefix_samples(
    tokenizer: object,
    samples: list[dict],
) -> list[dict]:
    """Prepare samples using only the raw FineWeb prefix (no chat template).

    Used for WCS audit so the target word is tested as the literal next token
    after the original passage, regardless of model variant.
    """
    prepared = []
    for sample in samples:
        prompt = sample["prefix"]
        token_ids = list(tokenizer.encode(prompt, add_special_tokens=True))
        transformed = dict(sample)
        transformed["prefix"] = prompt
        transformed["_prefix_token_ids"] = token_ids
        transformed["context_token_count"] = len(token_ids)
        prepared.append(transformed)
    return prepared


def _chat_prefix_samples(
    tokenizer: object,
    model_spec: ModelSpec,
    samples: list[dict],
    requested_words: int,
) -> list[dict]:
    """Prepare samples using a chat template with continuation instruction.

    Used for open generation so instruct/distill models produce coherent text.
    """
    if not hasattr(tokenizer, "apply_chat_template"):
        raise TypeError(f"{model_spec.model_id} tokenizer has no chat template support")
    prepared = []
    for sample in samples:
        messages = [
            {
                "role": "user",
                "content": continuation_instruction(sample["prefix"], requested_words),
            }
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        token_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
        if isinstance(token_ids, Mapping):
            token_ids = token_ids["input_ids"]
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        if token_ids and isinstance(token_ids[0], (list, tuple)):
            if len(token_ids) != 1:
                raise ValueError("Expected one chat-template input sequence")
            token_ids = token_ids[0]
        transformed = dict(sample)
        transformed["prefix"] = str(prompt)
        transformed["_prefix_token_ids"] = [int(tid) for tid in token_ids]
        transformed["context_token_count"] = len(token_ids)
        prepared.append(transformed)
    return prepared


def prepared_samples_for_audit(
    tokenizer: object,
    samples: list[dict],
) -> tuple[list[dict], str]:
    """Always use raw prefix for WCS audit — measures next-token reachability."""
    return _raw_prefix_samples(tokenizer, samples), "raw_continuation"


def prepared_samples_for_generation(
    tokenizer: object,
    model_spec: ModelSpec,
    samples: list[dict],
    requested_words: int,
) -> tuple[list[dict], str]:
    """Use chat template for instruct/distill, raw prefix for base models."""
    if _is_chat_model(model_spec):
        return _chat_prefix_samples(tokenizer, model_spec, samples, requested_words), "chat_continuation_instruction"
    return _raw_prefix_samples(tokenizer, samples), "raw_continuation"


def generation_key(row: dict) -> tuple[str, float, str, float]:
    return (
        str(row["sample_id"]),
        float(row["temperature"]),
        str(row["decoder"]),
        float(row["parameter"]),
    )


def completed_generation_keys(path: Path) -> set[tuple[str, float, str, float]]:
    if not path.exists():
        return set()
    keys = set()
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                row = json.loads(line)
                if row.get("complete"):
                    keys.add(generation_key(row))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
    return keys


def deterministic_seed(base_seed: int, model_slug: str, sample_id: str, temperature: float) -> int:
    # Deliberately omit the decoder condition so paired configurations begin
    # from the same pseudorandom stream for a given model/context/temperature.
    raw = f"{base_seed}|{model_slug}|{sample_id}|{temperature:g}".encode()
    digest = hashlib.sha256(raw).digest()
    return (base_seed + int.from_bytes(digest[:4], "big")) % (2**31)


def write_conditioned_audit(
    model: object,
    tokenizer: object,
    model_spec: ModelSpec,
    samples: list[dict],
    temperatures: list[float],
    output: Path,
    prompt_mode: str,
) -> None:
    expected = len(samples) * len(temperatures)
    existing: dict[tuple[str, float], list[dict]] = defaultdict(list)
    if output.exists():
        with output.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                    existing[(str(row["sample_id"]), float(row["temperature"]))].append(row)
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    continue
    complete = {
        key
        for key, rows in existing.items()
        if len(rows) == int(rows[0].get("word_token_count", 0))
        and {int(row["word_token_index"]) for row in rows} == set(range(len(rows)))
    }
    if len(complete) == expected:
        print(f"[skip-audit] {model_spec.slug}: {expected} prompt/temperature paths complete", flush=True)
        return

    # Drop malformed/partial groups before resuming. Otherwise appending a
    # complete retry after a partial path would leave duplicate token rows and
    # make every later resume regard that path as incomplete.
    if output.exists():
        temporary = output.with_suffix(output.suffix + ".resume-tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            for key in sorted(complete):
                for row in sorted(
                    existing[key], key=lambda value: int(value["word_token_index"])
                ):
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        temporary.replace(output)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        for index, sample in enumerate(samples, 1):
            missing_temperatures = [
                temperature
                for temperature in temperatures
                if (str(sample["id"]), temperature) not in complete
            ]
            if not missing_temperatures:
                continue
            rows_by_temperature = audit_sample_temperatures(
                model=model,
                tokenizer=tokenizer,
                sample=sample,
                model_name=model_spec.model_id,
                temperatures=missing_temperatures,
            )
            for temperature, rows in rows_by_temperature.items():
                for row in rows:
                    payload = asdict(row)
                    payload["prompt_mode"] = prompt_mode
                    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                handle.flush()
            print(f"[audit] {model_spec.slug}: {index}/{len(samples)} contexts", flush=True)


def generate_model(
    model: object,
    tokenizer: object,
    model_spec: ModelSpec,
    samples: list[dict],
    temperatures: list[float],
    conditions: list[Condition],
    args: argparse.Namespace,
    prompt_mode: str,
) -> Path:
    import torch

    output = args.results_dir / "generations" / f"generation.{model_spec.slug}.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    done = completed_generation_keys(output) if args.resume else set()
    if not args.resume:
        output.write_text("", encoding="utf-8")
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_device = args.device if not args.device_map else str(next(model.parameters()).device)
    total = len(samples) * len(temperatures) * len(conditions)
    completed = len(done)
    print(f"[generate] {model_spec.slug}: {completed}/{total} already complete", flush=True)

    for temperature in temperatures:
        for condition in conditions:
            for sample in samples:
                key = (str(sample["id"]), temperature, condition.decoder, condition.parameter)
                if key in done:
                    continue
                seed = deterministic_seed(args.seed, model_spec.slug, str(sample["id"]), temperature)
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                prompt_ids = sample["_prefix_token_ids"][-args.max_input_tokens :]
                input_ids = torch.tensor(
                    [prompt_ids], dtype=torch.long, device=model_device
                )
                encoded = {
                    "input_ids": input_ids,
                    "attention_mask": torch.ones_like(input_ids),
                }
                input_width = len(prompt_ids)
                kwargs = condition.generation_kwargs()
                with torch.inference_mode():
                    generated = model.generate(
                        **encoded,
                        do_sample=True,
                        temperature=temperature,
                        max_new_tokens=args.max_new_tokens,
                        pad_token_id=tokenizer.pad_token_id,
                        **kwargs,
                    )
                generated_ids = generated[0, input_width:]
                raw_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
                all_words = lexical_tokens(raw_text)
                scored_words = all_words[: args.words]
                reached_target = len(all_words) >= args.words
                eos_ids = tokenizer.eos_token_id
                if eos_ids is None:
                    eos_set: set[int] = set()
                elif isinstance(eos_ids, int):
                    eos_set = {eos_ids}
                else:
                    eos_set = {int(value) for value in eos_ids}
                stopped_at_eos = any(int(token_id) in eos_set for token_id in generated_ids)
                row = {
                    "sample_id": sample["id"],
                    "model": model_spec.model_id,
                    "model_slug": model_spec.slug,
                    "model_variant": model_spec.variant,
                    "prompt_mode": prompt_mode,
                    "target_word": sample["word"],
                    "source_path": sample["source_path"],
                    "requested_words": args.words,
                    "generated_word_count": len(all_words),
                    "scored_word_count": len(scored_words),
                    "reached_target_words": reached_target,
                    "generated_token_count": int(generated_ids.numel()),
                    "stopped_at_eos": stopped_at_eos,
                    "temperature": temperature,
                    "decoder": condition.decoder,
                    "parameter": condition.parameter,
                    "seed": seed,
                    "ttr": type_token_ratio(scored_words),
                    "mtld": mtld(scored_words, threshold=args.mtld_threshold),
                    "text": " ".join(scored_words),
                    "complete": True,
                }
                with output.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                completed += 1
                if completed % 25 == 0 or completed == total:
                    print(f"[generate] {model_spec.slug}: {completed}/{total}", flush=True)
    return output


def confidence_interval(values: list[float]) -> tuple[float, float]:
    if len(values) < 2:
        value = values[0] if values else 0.0
        return value, value
    mean = fmean(values)
    standard_error = pstdev(values) / math.sqrt(len(values))
    return mean - 1.96 * standard_error, mean + 1.96 * standard_error


def summarize_generations(paths: list[Path], output: Path) -> list[dict]:
    groups: dict[tuple[str, str, float, str, float], list[dict]] = defaultdict(list)
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                key = (
                    row["model"],
                    row["model_slug"],
                    float(row["temperature"]),
                    row["decoder"],
                    float(row["parameter"]),
                )
                groups[key].append(row)
    summaries = []
    for key, rows in sorted(groups.items()):
        model, slug, temperature, decoder, parameter = key
        ttrs = [float(row["ttr"]) for row in rows if row["reached_target_words"]]
        mtlds = [float(row["mtld"]) for row in rows if row["reached_target_words"]]
        ttr_low, ttr_high = confidence_interval(ttrs)
        mtld_low, mtld_high = confidence_interval(mtlds)
        summaries.append(
            {
                "model": model,
                "model_slug": slug,
                "temperature": temperature,
                "decoder": decoder,
                "parameter": parameter,
                "contexts": len(rows),
                "contexts_reaching_target": len(ttrs),
                "completion_rate": len(ttrs) / len(rows) if rows else 0.0,
                "mean_ttr": fmean(ttrs) if ttrs else 0.0,
                "median_ttr": median(ttrs) if ttrs else 0.0,
                "sd_ttr": pstdev(ttrs) if ttrs else 0.0,
                "ttr_ci95_low": ttr_low,
                "ttr_ci95_high": ttr_high,
                "mean_mtld": fmean(mtlds) if mtlds else 0.0,
                "median_mtld": median(mtlds) if mtlds else 0.0,
                "sd_mtld": pstdev(mtlds) if mtlds else 0.0,
                "mtld_ci95_low": mtld_low,
                "mtld_ci95_high": mtld_high,
            }
        )
    write_csv(output, summaries)
    return summaries


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        if fieldnames:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    print(f"[summary] wrote {path}", flush=True)


def pearson(values_x: list[float], values_y: list[float]) -> float:
    mean_x = fmean(values_x)
    mean_y = fmean(values_y)
    centered_x = [value - mean_x for value in values_x]
    centered_y = [value - mean_y for value in values_y]
    denominator = math.sqrt(
        sum(value * value for value in centered_x)
        * sum(value * value for value in centered_y)
    )
    if denominator == 0:
        return float("nan")
    return sum(x * y for x, y in zip(centered_x, centered_y)) / denominator


def average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    index = 0
    while index < len(order):
        end = index + 1
        while end < len(order) and values[order[end]] == values[order[index]]:
            end += 1
        average = (index + 1 + end) / 2.0
        for position in range(index, end):
            ranks[order[position]] = average
        index = end
    return ranks


def permutation_correlation(
    values_x: list[float],
    values_y: list[float],
    *,
    ranked: bool,
    permutations: int = 10_000,
    seed: int = 13,
) -> tuple[float, float]:
    xs = average_ranks(values_x) if ranked else values_x
    ys = average_ranks(values_y) if ranked else values_y
    observed = pearson(xs, ys)
    if math.isnan(observed):
        return observed, float("nan")
    rng = random.Random(seed)
    extreme = 0
    permuted = list(ys)
    for _ in range(permutations):
        rng.shuffle(permuted)
        statistic = pearson(xs, permuted)
        if not math.isnan(statistic) and abs(statistic) >= abs(observed) - 1e-12:
            extreme += 1
    return observed, (extreme + 1) / (permutations + 1)


def add_correlations(wcs_path: Path, diversity_rows: list[dict], output: Path) -> None:

    with wcs_path.open("r", encoding="utf-8") as handle:
        wcs_rows = list(csv.DictReader(handle))
    wcs = {
        (
            row["model"],
            float(row["temperature"]),
            row["decoder"],
            float(row["parameter"]),
        ): float(row["wcs"])
        for row in wcs_rows
    }
    joined = []
    for row in diversity_rows:
        if row["decoder"] == "untruncated":
            continue
        key = (
            row["model"],
            float(row["temperature"]),
            row["decoder"],
            float(row["parameter"]),
        )
        if key in wcs:
            joined.append({**row, "wcs": wcs[key]})
    correlation_rows = []
    groupings: dict[tuple[str, float, str], list[dict]] = defaultdict(list)
    for row in joined:
        groupings[(row["model"], row["temperature"], "all_filters")].append(row)
        groupings[(row["model"], row["temperature"], row["decoder"])].append(row)
    for (model, temperature, decoder), rows in sorted(groupings.items()):
        if len(rows) < 3:
            continue
        xs = [row["wcs"] for row in rows]
        for metric in ("mean_ttr", "median_ttr", "mean_mtld", "median_mtld"):
            ys = [float(row[metric]) for row in rows]
            pearson_r, pearson_p = permutation_correlation(xs, ys, ranked=False)
            spearman_rho, spearman_p = permutation_correlation(xs, ys, ranked=True)
            correlation_rows.append(
                {
                    "model": model,
                    "temperature": temperature,
                    "decoder": decoder,
                    "metric": metric,
                    "configurations": len(rows),
                    "pearson_r": pearson_r,
                    "pearson_p": pearson_p,
                    "spearman_rho": spearman_rho,
                    "spearman_p": spearman_p,
                    "p_value_method": "two_sided_permutation_10000",
                }
            )
    write_csv(output, correlation_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--models", default=None)
    parser.add_argument("--contexts", type=int, default=50)
    parser.add_argument("--words", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-input-tokens", type=int, default=1024)
    parser.add_argument("--temperatures", default="0.7,1.0")
    parser.add_argument("--top-k-values", default=",".join(map(str, DEFAULT_TOP_K)))
    parser.add_argument("--top-p-values", default=",".join(map(str, DEFAULT_TOP_P)))
    parser.add_argument("--min-p-values", default=",".join(map(str, DEFAULT_MIN_P)))
    parser.add_argument("--include-untruncated", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mtld-threshold", type=float, default=0.72)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--max-memory", default=None)
    parser.add_argument("--offload-folder", type=Path, default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--wcs-only", action="store_true", default=False,
        help="Run only the WCS forced-path audit (skip open generation).",
    )
    mode.add_argument(
        "--generation-only", action="store_true", default=False,
        help="Run only the open generation (skip WCS audit).",
    )
    return parser.parse_args()


def main() -> None:
    import torch

    args = parse_args()
    args.samples = args.samples.resolve()
    args.results_dir = args.results_dir.resolve()
    temperatures = parse_temperature_list(args.temperatures)
    conditions = conditions_from_args(args)
    samples = select_contexts(load_samples(args.samples), args.contexts, args.seed)
    models = select_models(args.models)
    run_wcs = not args.generation_only
    run_gen = not args.wcs_only
    phase_label = "wcs-only" if args.wcs_only else "generation-only" if args.generation_only else "full"
    print(
        f"[start] {len(models)} models, {len(samples)} contexts, "
        f"{len(temperatures)} temperatures, {len(conditions)} conditions "
        f"(mode={phase_label})",
        flush=True,
    )
    audit_paths = []
    generation_paths = []
    for model_spec in models:
        model, tokenizer = load_hf_model_and_tokenizer(
            model_spec.model_id,
            device=args.device,
            dtype=args.dtype,
            trust_remote_code=args.trust_remote_code,
            device_map=args.device_map,
            max_memory=parse_max_memory(args.max_memory),
            offload_folder=str(args.offload_folder) if args.offload_folder else None,
        )

        audit_path = args.results_dir / "conditioned_wcs" / f"audit.{model_spec.slug}.jsonl"
        gen_path = args.results_dir / "generations" / f"generation.{model_spec.slug}.jsonl"

        if run_wcs:
            # WCS audit: always raw prefix so the target word is the literal next token
            audit_samples, audit_prompt_mode = prepared_samples_for_audit(
                tokenizer, samples
            )
            write_conditioned_audit(
                model,
                tokenizer,
                model_spec,
                audit_samples,
                temperatures,
                audit_path,
                audit_prompt_mode,
            )
        if audit_path.exists():
            audit_paths.append(audit_path)

        if run_gen:
            # Generation: chat template for instruct/distill, raw prefix for base
            gen_samples, gen_prompt_mode = prepared_samples_for_generation(
                tokenizer, model_spec, samples, args.words
            )
            gen_path = generate_model(
                model,
                tokenizer,
                model_spec,
                gen_samples,
                temperatures,
                conditions,
                args,
                gen_prompt_mode,
            )
        if gen_path.exists():
            generation_paths.append(gen_path)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    wcs_summary = args.results_dir / "conditioned_wcs" / "wcs_summary.csv"
    if audit_paths:
        write_summary_csv(summarize_wcs(audit_paths), wcs_summary)
    if generation_paths:
        diversity_summary = args.results_dir / "lexical_diversity_by_config.csv"
        diversity_rows = summarize_generations(generation_paths, diversity_summary)
        if wcs_summary.exists():
            add_correlations(
                wcs_summary,
                diversity_rows,
                args.results_dir / "wcs_diversity_correlations.csv",
            )
    print(f"[done] results={args.results_dir}", flush=True)


if __name__ == "__main__":
    main()
