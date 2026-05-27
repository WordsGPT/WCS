#!/usr/bin/env python
"""Plot WCS against target-word token counts for an existing audit set."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
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


def load_word_metadata(samples_path: Path) -> dict[str, dict[str, object]]:
    metadata: dict[str, dict[str, object]] = {}
    order = 0
    with samples_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            word = str(row["word"])
            if word not in metadata:
                order += 1
                metadata[word] = {
                    "sample_order": order,
                    "rank": int(row["rank"]),
                    "count": row.get("count"),
                    "sample_contexts": 0,
                }
            metadata[word]["sample_contexts"] = int(metadata[word]["sample_contexts"]) + 1
    return metadata


def selected_words(metadata: dict[str, dict[str, object]], word_limit: int) -> set[str]:
    words = [
        word
        for word, _row in sorted(
            metadata.items(),
            key=lambda item: int(item[1]["sample_order"]),
        )
    ]
    if word_limit > 0:
        words = words[:word_limit]
    return set(words)


def load_context_groups(
    audit_paths: Iterable[Path],
    allowed_words: set[str],
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
                word = str(row["word"])
                model = str(row["model"])
                if word not in allowed_words:
                    continue
                if include_models is not None and model not in include_models:
                    continue
                if model in exclude_models:
                    continue
                key = (
                    model,
                    float(row.get("temperature", 1.0)),
                    word,
                    str(row["sample_id"]),
                )
                groups[key].append(row)
    for rows in groups.values():
        rows.sort(key=lambda row: int(row["word_token_index"]))
    return groups


def context_score(rows: list[dict], top_k: list[int], top_p: list[float], min_p: list[float]) -> float:
    checks: list[bool] = []
    checks.extend(word_survives_top_k(rows, k) for k in top_k)
    checks.extend(word_survives_top_p(rows, p) for p in top_p)
    checks.extend(word_survives_min_p(rows, p) for p in min_p)
    if not checks:
        return 0.0
    return sum(1 for check in checks if check) / len(checks)


def summarize_tokenization(
    metadata: dict[str, dict[str, object]],
    groups: dict[tuple[str, float, str, str], list[dict]],
    top_k: list[int],
    top_p: list[float],
    min_p: list[float],
) -> list[dict[str, object]]:
    totals: dict[tuple[str, float, str], dict[str, float]] = defaultdict(
        lambda: {
            "wcs_sum": 0.0,
            "contexts": 0.0,
            "token_count_sum": 0.0,
        }
    )
    for (model, temperature, word, _sample_id), rows in groups.items():
        if word not in metadata:
            continue
        key = (model, temperature, word)
        totals[key]["wcs_sum"] += context_score(rows, top_k, top_p, min_p)
        totals[key]["contexts"] += 1
        totals[key]["token_count_sum"] += float(rows[0].get("word_token_count", len(rows)))

    rows: list[dict[str, object]] = []
    for (model, temperature, word), values in sorted(totals.items()):
        word_metadata = metadata[word]
        contexts = values["contexts"]
        rows.append(
            {
                "model": model,
                "temperature": temperature,
                "word": word,
                "sample_order": word_metadata["sample_order"],
                "rank": word_metadata["rank"],
                "count": word_metadata["count"],
                "sample_contexts": word_metadata["sample_contexts"],
                "evaluated_contexts": int(contexts),
                "mean_wcs": values["wcs_sum"] / contexts if contexts else 0.0,
                "mean_token_count": values["token_count_sum"] / contexts if contexts else 0.0,
            }
        )
    return rows


def add_average_token_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
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
                "sample_order": first["sample_order"],
                "rank": first["rank"],
                "count": first["count"],
                "sample_contexts": first["sample_contexts"],
                "evaluated_contexts": sum(int(row["evaluated_contexts"]) for row in word_rows),
                "mean_wcs": sum(float(row["mean_wcs"]) for row in word_rows) / len(word_rows),
                "mean_token_count": sum(float(row["mean_token_count"]) for row in word_rows) / len(word_rows),
            }
        )
    return rows + aggregate_rows


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


def correlation_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["model"]), str(row["temperature"]))].append(row)

    output: list[dict[str, object]] = []
    for (model, temperature), model_rows in sorted(grouped.items()):
        token_counts = [float(row["mean_token_count"]) for row in model_rows]
        mean_wcs = [float(row["mean_wcs"]) for row in model_rows]
        output.append(
            {
                "model": model,
                "temperature": temperature,
                "n_words": len(model_rows),
                "mean_token_count": sum(token_counts) / len(token_counts) if token_counts else "",
                "mean_wcs": sum(mean_wcs) / len(mean_wcs) if mean_wcs else "",
                "token_count_pearson": pearson(token_counts, mean_wcs),
                "token_count_spearman": spearman(token_counts, mean_wcs),
            }
        )
    return output


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")
    return slug or "model"


def plot_rows(rows: list[dict[str, object]], plot_dir: Path, label: str) -> list[Path]:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for plotting") from exc

    plot_dir.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []

    aggregate_rows = [row for row in rows if str(row["model"]) == "ALL_MODELS"]
    if aggregate_rows:
        output_path = plot_dir / f"wcs_vs_avg_token_count.{label}.png"
        make_scatter(
            plt,
            aggregate_rows,
            output_path,
            title=f"WCS vs average token count ({label.replace('_', ' ')})",
            x_label="Average target-token count across models",
        )
        output_paths.append(output_path)

    for model in sorted({str(row["model"]) for row in rows if str(row["model"]) != "ALL_MODELS"}):
        model_rows = [row for row in rows if str(row["model"]) == model]
        output_path = plot_dir / f"wcs_vs_token_count.{slugify(model)}.{label}.png"
        make_scatter(
            plt,
            model_rows,
            output_path,
            title=f"WCS vs token count: {model}",
            x_label="Target-token count",
        )
        output_paths.append(output_path)

    return output_paths


def make_scatter(plt: object, rows: list[dict[str, object]], output_path: Path, title: str, x_label: str) -> None:
    xs = [float(row["mean_token_count"]) for row in rows]
    ys = [float(row["mean_wcs"]) for row in rows]
    ranks = [float(row["rank"]) for row in rows]
    correlation = spearman(xs, ys)

    plt.figure(figsize=(7.5, 5.2))
    scatter = plt.scatter(xs, ys, c=ranks, cmap="viridis_r", alpha=0.78, s=34, edgecolors="none")
    plt.xlabel(x_label)
    plt.ylabel("Mean WCS")
    plt.ylim(-0.02, 1.02)
    if xs:
        min_x = min(xs)
        max_x = max(xs)
        if min_x == max_x:
            plt.xlim(min_x - 0.5, max_x + 0.5)
        else:
            margin = max(0.2, (max_x - min_x) * 0.08)
            plt.xlim(min_x - margin, max_x + margin)
    subtitle = "Spearman rho=N/A" if correlation is None else f"Spearman rho={correlation:.3f}"
    plt.title(f"{title}\n{subtitle}")
    colorbar = plt.colorbar(scatter)
    colorbar.set_label("Frequency rank")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def write_csv(rows: list[dict[str, object]], output_path: Path, fieldnames: list[str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="Audit JSONL files or directories containing them.")
    parser.add_argument("--samples", type=Path, default=ROOT / "data/processed/samples.jsonl")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/tokenization_wcs_100")
    parser.add_argument("--word-limit", type=int, default=100, help="Use the first N sampled target words; 0 means all words.")
    parser.add_argument("--top-k", default=",".join(str(value) for value in DEFAULT_TOP_K))
    parser.add_argument("--top-p", default=",".join(str(value) for value in DEFAULT_TOP_P))
    parser.add_argument("--min-p", default=",".join(str(value) for value in DEFAULT_MIN_P))
    parser.add_argument("--include-model", action="append", default=None)
    parser.add_argument("--exclude-model", action="append", default=[])
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit_paths = iter_audit_paths(args.paths)
    if not audit_paths:
        raise SystemExit("No audit*.jsonl files found in the provided paths.")

    metadata = load_word_metadata(args.samples)
    words = selected_words(metadata, args.word_limit)
    groups = load_context_groups(
        audit_paths,
        allowed_words=words,
        include_models=set(args.include_model) if args.include_model else None,
        exclude_models=set(args.exclude_model),
    )
    per_model_rows = summarize_tokenization(
        metadata=metadata,
        groups=groups,
        top_k=parse_number_list(args.top_k, int),
        top_p=parse_number_list(args.top_p, float),
        min_p=parse_number_list(args.min_p, float),
    )
    rows = add_average_token_rows(per_model_rows)
    correlations = correlation_rows(rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    word_label = f"first_{args.word_limit}_words" if args.word_limit > 0 else "all_words"
    per_word_output = args.output_dir / f"tokenization_wcs.{word_label}.csv"
    correlation_output = args.output_dir / f"tokenization_wcs_correlations.{word_label}.csv"
    write_csv(
        rows,
        per_word_output,
        [
            "model",
            "temperature",
            "word",
            "sample_order",
            "rank",
            "count",
            "sample_contexts",
            "evaluated_contexts",
            "mean_wcs",
            "mean_token_count",
        ],
    )
    write_csv(
        correlations,
        correlation_output,
        [
            "model",
            "temperature",
            "n_words",
            "mean_token_count",
            "mean_wcs",
            "token_count_pearson",
            "token_count_spearman",
        ],
    )

    print(f"Audit files: {len(audit_paths)}")
    print(f"Selected words: {len(words)}")
    print(f"Per-model word rows: {len(per_model_rows)}")
    print(f"Wrote tokenization WCS: {per_word_output}")
    print(f"Wrote tokenization correlations: {correlation_output}")
    if not args.no_plots:
        for output_path in plot_rows(rows, args.output_dir / "plots", word_label):
            print(f"Wrote plot: {output_path}")


if __name__ == "__main__":
    main()
