#!/usr/bin/env python
"""Build JSON data assets for the four reviewer-facing static sites."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


MODEL_META = {
    "google/gemma-3-12b-pt": ("gemma3-12b-base", "Gemma 3 12B Base"),
    "google/gemma-3-12b-it": ("gemma3-12b-it", "Gemma 3 12B Instruct"),
    "meta-llama/Llama-3.1-8B": ("llama31-8b-base", "Llama 3.1 8B Base"),
    "meta-llama/Llama-3.1-8B-Instruct": (
        "llama31-8b-instruct",
        "Llama 3.1 8B Instruct",
    ),
    "mistralai/Mistral-7B-v0.3": ("mistral7b-v03-base", "Mistral 7B Base"),
    "mistralai/Mistral-7B-Instruct-v0.3": (
        "mistral7b-v03-instruct",
        "Mistral 7B Instruct",
    ),
    "Qwen/Qwen2.5-14B": ("qwen25-14b-base", "Qwen 2.5 14B Base"),
    "Qwen/Qwen2.5-14B-Instruct": ("qwen25-14b-instruct", "Qwen 2.5 14B Instruct"),
    "Qwen/Qwen3.5-9B-Base": ("qwen35-9b-base", "Qwen 3.5 9B Base"),
    "Qwen/Qwen3.5-9B": ("qwen35-9b-instruct", "Qwen 3.5 9B Instruct"),
    "google/gemma-4-E4B": ("gemma4-e4b-base", "Gemma 4 E4B Base"),
    "google/gemma-4-E4B-it": ("gemma4-e4b-it", "Gemma 4 E4B Instruct"),
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B": (
        "deepseek-qwen14b-distill",
        "DeepSeek R1 Distill Qwen 14B",
    ),
}

TOP_K = tuple(range(1, 21)) + tuple(range(25, 81, 5))
TOP_P = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99)
MIN_P = (0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10)


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"[write] {path}")


def audit_groups(path: Path) -> tuple[str, dict[str, list[dict]]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    model = ""
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            model = str(row["model"])
            groups[str(row["sample_id"])].append(row)
    if not groups:
        raise ValueError(f"Empty audit file: {path}")
    return model, groups


def survives(rows: list[dict], decoder: str, parameter: float) -> bool:
    if decoder == "top_k":
        return all(int(row["rank"]) <= int(parameter) for row in rows)
    if decoder == "top_p":
        return all(
            float(row["cumulative_probability"]) - float(row["probability"])
            <= parameter
            for row in rows
        )
    if decoder == "min_p":
        return all(float(row["probability_ratio_to_top"]) >= parameter for row in rows)
    raise ValueError(decoder)


def percentile_ci(word_means: np.ndarray, rng: np.random.Generator, draws: int) -> tuple[float, float]:
    if word_means.size <= 1:
        value = float(word_means.mean()) if word_means.size else 0.0
        return value, value
    indices = rng.integers(0, word_means.size, size=(draws, word_means.size))
    bootstrap = word_means[indices].mean(axis=1)
    low, high = np.quantile(bootstrap, (0.025, 0.975))
    return float(low), float(high)


def build_wcs_assets(
    audit_dir: Path,
    output_dir: Path,
    bootstrap_draws: int,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    overall_rows = []
    stratum_rows = []
    audit_paths = sorted(audit_dir.glob("audit.*.jsonl"))
    if len(audit_paths) != 6:
        raise ValueError(f"Expected six full WCS audit files in {audit_dir}; found {len(audit_paths)}")
    schedules = {"top_k": TOP_K, "top_p": TOP_P, "min_p": MIN_P}

    for path in audit_paths:
        model, groups = audit_groups(path)
        slug, label = MODEL_META[model]
        samples = []
        for sample_id, rows in groups.items():
            rows.sort(key=lambda row: int(row["word_token_index"]))
            samples.append(
                {
                    "sample_id": sample_id,
                    "word": str(rows[0]["word"]),
                    "token_count": int(rows[0]["word_token_count"]),
                    "rows": rows,
                }
            )
        words = sorted({sample["word"] for sample in samples})
        word_index = {word: index for index, word in enumerate(words)}
        contexts_by_word: dict[str, list[int]] = defaultdict(list)
        for index, sample in enumerate(samples):
            contexts_by_word[sample["word"]].append(index)

        for decoder, parameters in schedules.items():
            for parameter in parameters:
                values = np.asarray(
                    [float(survives(sample["rows"], decoder, float(parameter))) for sample in samples]
                )
                per_word = np.asarray(
                    [values[contexts_by_word[word]].mean() for word in words], dtype=float
                )
                low, high = percentile_ci(per_word, rng, bootstrap_draws)
                overall_rows.append(
                    {
                        "model": model,
                        "model_slug": slug,
                        "model_label": label,
                        "decoder": decoder,
                        "parameter": float(parameter),
                        "wcs": float(values.mean()),
                        "ci95_low": low,
                        "ci95_high": high,
                        "contexts": len(samples),
                        "words": len(words),
                        "ci_method": f"target-word cluster bootstrap, {bootstrap_draws} draws",
                    }
                )

                token_counts = sorted({sample["token_count"] for sample in samples})
                for token_count in token_counts:
                    selected_words = sorted(
                        {
                            sample["word"]
                            for sample in samples
                            if sample["token_count"] == token_count
                        }
                    )
                    selected_indices = [
                        index
                        for index, sample in enumerate(samples)
                        if sample["token_count"] == token_count
                    ]
                    selected_word_means = np.asarray(
                        [values[contexts_by_word[word]].mean() for word in selected_words],
                        dtype=float,
                    )
                    stratum_low, stratum_high = percentile_ci(
                        selected_word_means, rng, bootstrap_draws
                    )
                    stratum_rows.append(
                        {
                            "model": model,
                            "model_slug": slug,
                            "model_label": label,
                            "decoder": decoder,
                            "parameter": float(parameter),
                            "token_count": token_count,
                            "wcs": float(values[selected_indices].mean()),
                            "ci95_low": stratum_low,
                            "ci95_high": stratum_high,
                            "contexts": len(selected_indices),
                            "words": len(selected_words),
                            "ci_method": f"target-word cluster bootstrap, {bootstrap_draws} draws",
                        }
                    )

    metadata = {
        "corpus": "HuggingFaceFW/fineweb sample-10BT",
        "sample_design": "200 target words × 50 contexts",
        "temperature": 1.0,
        "models": [
            {"model": model, "slug": MODEL_META[model][0], "label": MODEL_META[model][1]}
            for model in sorted({row["model"] for row in overall_rows})
        ],
        "ci_method": f"target-word cluster bootstrap, {bootstrap_draws} draws, percentile 95% interval",
        "top_p_definition": "boundary token retained; survives when mass strictly above target is at most p",
    }
    write_json(output_dir / "wcs_all.json", {"metadata": metadata, "rows": overall_rows})
    write_json(output_dir / "wcs_token_strata.json", {"metadata": metadata, "rows": stratum_rows})


def build_downstream_asset(results_dir: Path, corrected_dir: Path, output_dir: Path) -> None:
    diversity = read_csv(results_dir / "lexical_diversity_by_config.csv")
    wcs = read_csv(results_dir / "conditioned_wcs" / "wcs_summary.csv")
    primary = read_csv(corrected_dir / "primary_correlations_holm.csv")
    paired = read_csv(corrected_dir / "paired_endpoint_effects_holm.csv")
    completion = read_csv(corrected_dir / "completion_rates.csv")
    descriptive = read_csv(corrected_dir / "sampler_correlations_descriptive.csv")
    pooled_path = corrected_dir / "pooled_context_effects.csv"
    pooled = read_csv(pooled_path) if pooled_path.exists() else []
    wcs_index = {
        (
            row["model"],
            float(row["temperature"]),
            row["decoder"],
            float(row["parameter"]),
        ): float(row["wcs"])
        for row in wcs
    }
    joined = []
    for row in diversity:
        if row["decoder"] == "untruncated":
            continue
        key = (
            row["model"],
            float(row["temperature"]),
            row["decoder"],
            float(row["parameter"]),
        )
        slug, label = MODEL_META[row["model"]]
        joined.append(
            {
                "model": row["model"],
                "model_slug": slug,
                "model_label": label,
                "temperature": float(row["temperature"]),
                "decoder": row["decoder"],
                "parameter": float(row["parameter"]),
                "wcs": wcs_index[key],
                "mean_ttr": float(row["mean_ttr"]),
                "mean_mtld": float(row["mean_mtld"]),
                "median_ttr": float(row["median_ttr"]),
                "median_mtld": float(row["median_mtld"]),
                "completion_rate": float(row["completion_rate"]),
                "contexts": int(row["contexts"]),
                "contexts_reaching_target": int(row["contexts_reaching_target"]),
            }
        )
    value = {
        "metadata": {
            "models": len({row["model"] for row in diversity}),
            "generations": sum(int(row["contexts"]) for row in diversity),
            "contexts": 50,
            "scoring_window": "first 100 lexical words",
            "temperatures": [0.7, 1.0],
            "prompting": "WCS uses raw FineWeb prefixes for all checkpoints; generation uses raw continuation for Base and native chat continuation instructions for post-trained checkpoints",
            "primary_family": f"{len(primary)} aggregate Spearman tests; Holm FWER correction",
            "secondary_family": f"{len(paired)} paired endpoint Wilcoxon tests; Holm FWER correction",
        },
        "rows": joined,
        "primary_correlations": primary,
        "paired_effects": paired,
        "completion": completion,
        "sampler_correlations": descriptive,
        "pooled_context_effects": pooled,
    }
    write_json(output_dir / "downstream.json", value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-audit-dir", type=Path, required=True)
    parser.add_argument("--downstream-results-dir", type=Path, required=True)
    parser.add_argument("--corrected-analysis-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=10298)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_wcs_assets(
        args.full_audit_dir.resolve(),
        args.output_dir.resolve(),
        args.bootstrap_draws,
        args.seed,
    )
    build_downstream_asset(
        args.downstream_results_dir.resolve(),
        args.corrected_analysis_dir.resolve(),
        args.output_dir.resolve(),
    )


if __name__ == "__main__":
    main()
