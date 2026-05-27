#!/usr/bin/env python
"""Summarize per-word WCS and correlations with frequency rank/count."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from wcs.metrics import (
    DEFAULT_MIN_P,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    parse_number_list,
    word_survives_min_p,
    word_survives_top_k,
    word_survives_top_p,
)


def iter_audit_paths(paths: Iterable[Path]) -> list[Path]:
    audit_paths: list[Path] = []
    for path in paths:
        if path.is_dir():
            audit_paths.extend(sorted(path.rglob("audit*.jsonl")))
        elif path.is_file():
            audit_paths.append(path)
    return audit_paths


def load_sample_metadata(samples_path: Path) -> dict[str, dict[str, object]]:
    metadata: dict[str, dict[str, object]] = {}
    with samples_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            word = str(row["word"])
            sample_metadata = row.get("metadata", {})
            count = row.get("count")
            if word not in metadata:
                metadata[word] = {
                    "rank": int(row["rank"]),
                    "count": int(count) if count is not None else None,
                    "sample_contexts": 0,
                    "frequency_band": str(sample_metadata.get("stratum", "")),
                }
            metadata[word]["sample_contexts"] = int(metadata[word]["sample_contexts"]) + 1
    return metadata


def load_context_groups(
    audit_paths: Iterable[Path],
    include_models: set[str] | None,
    exclude_models: set[str],
) -> dict[tuple[str, float, str, str], list[dict]]:
    groups: dict[tuple[str, float, str, str], list[dict]] = defaultdict(list)
    for path in audit_paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                model = str(row["model"])
                if include_models is not None and model not in include_models:
                    continue
                if model in exclude_models:
                    continue
                key = (
                    model,
                    float(row.get("temperature", 1.0)),
                    str(row["word"]),
                    str(row["sample_id"]),
                )
                groups[key].append(row)
    for rows in groups.values():
        rows.sort(key=lambda row: int(row["word_token_index"]))
    return groups


def context_checks(rows: list[dict], top_k: list[int], top_p: list[float], min_p: list[float]) -> list[bool]:
    checks: list[bool] = []
    checks.extend(word_survives_top_k(rows, k) for k in top_k)
    checks.extend(word_survives_top_p(rows, p) for p in top_p)
    checks.extend(word_survives_min_p(rows, p) for p in min_p)
    return checks


def summarize_words(
    metadata: dict[str, dict[str, object]],
    groups: dict[tuple[str, float, str, str], list[dict]],
    top_k: list[int],
    top_p: list[float],
    min_p: list[float],
) -> list[dict[str, object]]:
    totals: dict[tuple[str, float, str], dict[str, float]] = defaultdict(
        lambda: {
            "reachable_settings": 0.0,
            "total_settings": 0.0,
            "contexts_with_any": 0.0,
            "evaluated_contexts": 0.0,
        }
    )
    for (model, temperature, word, _sample_id), rows in groups.items():
        if word not in metadata:
            continue
        checks = context_checks(rows, top_k, top_p, min_p)
        if not checks:
            continue
        total = totals[(model, temperature, word)]
        reachable = sum(1 for check in checks if check)
        total["reachable_settings"] += reachable
        total["total_settings"] += len(checks)
        total["contexts_with_any"] += 1 if reachable else 0
        total["evaluated_contexts"] += 1

    rows: list[dict[str, object]] = []
    for (model, temperature, word), values in sorted(totals.items()):
        word_metadata = metadata[word]
        total_settings = values["total_settings"]
        evaluated_contexts = values["evaluated_contexts"]
        rows.append(
            {
                "model": model,
                "temperature": temperature,
                "word": word,
                "rank": word_metadata["rank"],
                "count": word_metadata["count"],
                "frequency_band": word_metadata["frequency_band"],
                "sample_contexts": word_metadata["sample_contexts"],
                "evaluated_contexts": int(evaluated_contexts),
                "mean_wcs": values["reachable_settings"] / total_settings if total_settings else 0.0,
                "any_reachable_rate": values["contexts_with_any"] / evaluated_contexts if evaluated_contexts else 0.0,
                "evaluated_context_settings": int(total_settings),
            }
        )
    return rows


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denom_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    denom_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if denom_x == 0 or denom_y == 0:
        return None
    return numerator / (denom_x * denom_y)


def average_ranks(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(indexed):
        end = index + 1
        while end < len(indexed) and indexed[end][1] == indexed[index][1]:
            end += 1
        average = (index + 1 + end) / 2
        for original_index, _value in indexed[index:end]:
            ranks[original_index] = average
        index = end
    return ranks


def spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    return pearson(average_ranks(xs), average_ranks(ys))


def add_aggregate_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_word: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_word[str(row["word"])].append(row)

    aggregate_rows: list[dict[str, object]] = []
    for word, word_rows in sorted(by_word.items()):
        first = word_rows[0]
        aggregate_rows.append(
            {
                "model": "ALL_MODELS",
                "temperature": "",
                "word": word,
                "rank": first["rank"],
                "count": first["count"],
                "frequency_band": first["frequency_band"],
                "sample_contexts": first["sample_contexts"],
                "evaluated_contexts": sum(int(row["evaluated_contexts"]) for row in word_rows),
                "mean_wcs": sum(float(row["mean_wcs"]) for row in word_rows) / len(word_rows),
                "any_reachable_rate": sum(float(row["any_reachable_rate"]) for row in word_rows) / len(word_rows),
                "evaluated_context_settings": sum(int(row["evaluated_context_settings"]) for row in word_rows),
            }
        )
    return rows + aggregate_rows


def correlation_rows(word_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_model_temperature: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in word_rows:
        by_model_temperature[(str(row["model"]), str(row["temperature"]))].append(row)

    rows: list[dict[str, object]] = []
    for (model, temperature), values in sorted(by_model_temperature.items()):
        ranks = [float(row["rank"]) for row in values]
        mean_wcs = [float(row["mean_wcs"]) for row in values]
        counts_and_wcs = [
            (float(row["count"]), float(row["mean_wcs"]))
            for row in values
            if row["count"] not in ("", None) and float(row["count"]) > 0
        ]
        log_counts = [math.log10(count) for count, _wcs in counts_and_wcs]
        count_wcs = [wcs for _count, wcs in counts_and_wcs]
        rows.append(
            {
                "model": model,
                "temperature": temperature,
                "n_words": len(values),
                "mean_wcs": sum(mean_wcs) / len(mean_wcs) if mean_wcs else "",
                "rank_pearson": pearson(ranks, mean_wcs),
                "rank_spearman": spearman(ranks, mean_wcs),
                "log_count_pearson": pearson(log_counts, count_wcs),
                "log_count_spearman": spearman(log_counts, count_wcs),
            }
        )
    return rows


def write_csv(rows: list[dict[str, object]], output_path: Path, fieldnames: list[str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="Audit JSONL files or directories containing them.")
    parser.add_argument("--samples", type=Path, default=ROOT / "data/processed/samples.frequency_1k.jsonl")
    parser.add_argument("--per-word-output", type=Path, default=ROOT / "results/frequency_wcs/per_word_wcs.csv")
    parser.add_argument("--correlation-output", type=Path, default=ROOT / "results/frequency_wcs/frequency_correlations.csv")
    parser.add_argument("--top-k", default=",".join(str(value) for value in DEFAULT_TOP_K))
    parser.add_argument("--top-p", default=",".join(str(value) for value in DEFAULT_TOP_P))
    parser.add_argument("--min-p", default=",".join(str(value) for value in DEFAULT_MIN_P))
    parser.add_argument("--include-model", action="append", default=None)
    parser.add_argument("--exclude-model", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit_paths = iter_audit_paths(args.paths)
    if not audit_paths:
        raise SystemExit("No audit*.jsonl files found in the provided paths.")

    metadata = load_sample_metadata(args.samples)
    groups = load_context_groups(
        audit_paths,
        include_models=set(args.include_model) if args.include_model else None,
        exclude_models=set(args.exclude_model),
    )
    per_model_rows = summarize_words(
        metadata=metadata,
        groups=groups,
        top_k=parse_number_list(args.top_k, int),
        top_p=parse_number_list(args.top_p, float),
        min_p=parse_number_list(args.min_p, float),
    )
    word_rows = add_aggregate_rows(per_model_rows)
    correlations = correlation_rows(word_rows)

    per_word_fields = [
        "model",
        "temperature",
        "word",
        "rank",
        "count",
        "frequency_band",
        "sample_contexts",
        "evaluated_contexts",
        "mean_wcs",
        "any_reachable_rate",
        "evaluated_context_settings",
    ]
    correlation_fields = [
        "model",
        "temperature",
        "n_words",
        "mean_wcs",
        "rank_pearson",
        "rank_spearman",
        "log_count_pearson",
        "log_count_spearman",
    ]
    write_csv(word_rows, args.per_word_output, per_word_fields)
    write_csv(correlations, args.correlation_output, correlation_fields)
    print(f"Audit files: {len(audit_paths)}")
    print(f"Per-model word rows: {len(per_model_rows)}")
    print(f"Wrote per-word WCS: {args.per_word_output}")
    print(f"Wrote correlations: {args.correlation_output}")


if __name__ == "__main__":
    main()
