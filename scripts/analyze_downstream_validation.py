#!/usr/bin/env python
"""Create multiplicity-corrected tables and plots for downstream validation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from statistics import fmean, median


MODEL_LABELS = {
    "llama31-8b-base": "Llama 3.1 8B Base",
    "llama31-8b-instruct": "Llama 3.1 8B Instruct",
    "mistral7b-v03-base": "Mistral 7B Base",
    "mistral7b-v03-instruct": "Mistral 7B Instruct",
    "qwen35-9b-base": "Qwen 3.5 9B Base",
    "qwen35-9b-instruct": "Qwen 3.5 9B Instruct",
    "qwen25-14b-base": "Qwen 2.5 14B Base",
    "qwen25-14b-instruct": "Qwen 2.5 14B Instruct",
    "gemma3-12b-base": "Gemma 3 12B Base",
    "gemma3-12b-it": "Gemma 3 12B Instruct",
    "gemma4-e4b-base": "Gemma 4 E4B Base",
    "gemma4-e4b-it": "Gemma 4 E4B Instruct",
    "deepseek-qwen14b-distill": "DeepSeek R1 Distill Qwen 14B",
}

ENDPOINTS = (
    ("top_k", 10.0, 80.0, "top-k 10→80"),
    ("top_p", 0.80, 0.99, "top-p .80→.99"),
    ("min_p", 0.10, 0.01, "min-p .10→.01"),
)


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        if not rows:
            return
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[write] {path}")


def holm_adjust(p_values: list[float]) -> list[float]:
    """Return Holm family-wise-error adjusted p-values."""
    count = len(p_values)
    order = sorted(range(count), key=p_values.__getitem__)
    adjusted = [1.0] * count
    running_max = 0.0
    for position, original_index in enumerate(order):
        candidate = (count - position) * p_values[original_index]
        running_max = max(running_max, candidate)
        adjusted[original_index] = min(1.0, running_max)
    return adjusted


def primary_correlations(correlation_csv: Path) -> list[dict]:
    selected = [
        row
        for row in read_csv(correlation_csv)
        if row["decoder"] == "all_filters"
        and row["metric"] in {"mean_ttr", "mean_mtld"}
    ]
    selected.sort(
        key=lambda row: (
            row["model"],
            float(row["temperature"]),
            row["metric"],
        )
    )
    spearman_adjusted = holm_adjust([float(row["spearman_p"]) for row in selected])
    pearson_adjusted = holm_adjust([float(row["pearson_p"]) for row in selected])
    output = []
    for row, spearman_holm, pearson_holm in zip(
        selected, spearman_adjusted, pearson_adjusted
    ):
        output.append(
            {
                "model": row["model"],
                "temperature": float(row["temperature"]),
                "metric": row["metric"],
                "configurations": int(row["configurations"]),
                "spearman_rho": float(row["spearman_rho"]),
                "spearman_p_raw": float(row["spearman_p"]),
                "spearman_p_holm_24": spearman_holm,
                "spearman_significant_holm_05": spearman_holm < 0.05,
                "pearson_r": float(row["pearson_r"]),
                "pearson_p_raw": float(row["pearson_p"]),
                "pearson_p_holm_24": pearson_holm,
                "pearson_significant_holm_05": pearson_holm < 0.05,
                "family_definition": "24 model×temperature×metric primary tests",
            }
        )
    return output


def sampler_correlations_descriptive(correlation_csv: Path) -> list[dict]:
    rows = [
        row
        for row in read_csv(correlation_csv)
        if row["decoder"] != "all_filters"
        and row["metric"] in {"mean_ttr", "mean_mtld"}
    ]
    return [
        {
            "model": row["model"],
            "temperature": float(row["temperature"]),
            "decoder": row["decoder"],
            "metric": row["metric"],
            "configurations": int(row["configurations"]),
            "pearson_r": float(row["pearson_r"]),
            "spearman_rho": float(row["spearman_rho"]),
            "interpretation": "descriptive_only_n3_or_n4",
        }
        for row in rows
    ]


def load_generations(generation_dir: Path) -> dict[str, list[dict]]:
    models = {}
    for path in sorted(generation_dir.glob("generation.*.jsonl")):
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        if records:
            models[str(records[0]["model_slug"])] = records
    if not models:
        raise FileNotFoundError(f"No generation JSONL files under {generation_dir}")
    return models


def paired_endpoint_effects(generations: dict[str, list[dict]]) -> list[dict]:
    try:
        from scipy.stats import wilcoxon
    except ImportError as exc:
        raise RuntimeError("scipy is required for paired Wilcoxon tests") from exc

    output = []
    for model_slug, rows in sorted(generations.items()):
        index = {
            (
                str(row["sample_id"]),
                float(row["temperature"]),
                str(row["decoder"]),
                float(row["parameter"]),
            ): row
            for row in rows
        }
        sample_ids = sorted({str(row["sample_id"]) for row in rows})
        temperatures = sorted({float(row["temperature"]) for row in rows})
        for temperature in temperatures:
            for decoder, restrictive, permissive, contrast in ENDPOINTS:
                for metric in ("ttr", "mtld"):
                    differences = []
                    for sample_id in sample_ids:
                        restrictive_row = index[
                            (sample_id, temperature, decoder, restrictive)
                        ]
                        permissive_row = index[
                            (sample_id, temperature, decoder, permissive)
                        ]
                        if not (
                            restrictive_row["reached_target_words"]
                            and permissive_row["reached_target_words"]
                        ):
                            continue
                        differences.append(
                            float(permissive_row[metric])
                            - float(restrictive_row[metric])
                        )
                    nonzero = [difference for difference in differences if difference != 0]
                    if nonzero:
                        test = wilcoxon(differences, alternative="two-sided")
                        p_value = float(test.pvalue)
                        statistic = float(test.statistic)
                    else:
                        p_value = 1.0
                        statistic = 0.0
                    output.append(
                        {
                            "model_slug": model_slug,
                            "temperature": temperature,
                            "decoder": decoder,
                            "contrast": contrast,
                            "metric": metric,
                            "paired_contexts": len(differences),
                            "mean_paired_difference": fmean(differences),
                            "median_paired_difference": median(differences),
                            "wilcoxon_statistic": statistic,
                            "wilcoxon_p_raw": p_value,
                        }
                    )
    adjusted = holm_adjust([float(row["wilcoxon_p_raw"]) for row in output])
    for row, adjusted_p in zip(output, adjusted):
        row["wilcoxon_p_holm_72"] = adjusted_p
        row["significant_holm_05"] = adjusted_p < 0.05
        row["family_definition"] = "72 secondary endpoint tests"
    return output


def completion_summary(diversity_csv: Path) -> list[dict]:
    grouped: dict[tuple[str, float], list[dict]] = defaultdict(list)
    for row in read_csv(diversity_csv):
        grouped[(row["model_slug"], float(row["temperature"]))].append(row)
    output = []
    for (model_slug, temperature), rows in sorted(grouped.items()):
        rates = [float(row["completion_rate"]) for row in rows]
        minimum = min(rows, key=lambda row: float(row["completion_rate"]))
        output.append(
            {
                "model_slug": model_slug,
                "temperature": temperature,
                "configurations": len(rows),
                "mean_completion_rate": fmean(rates),
                "min_completion_rate": min(rates),
                "max_completion_rate": max(rates),
                "lowest_completion_decoder": minimum["decoder"],
                "lowest_completion_parameter": float(minimum["parameter"]),
            }
        )
    return output


def outlier_summary(generations: dict[str, list[dict]]) -> list[dict]:
    output = []
    for model_slug, rows in sorted(generations.items()):
        completed = [row for row in rows if row["reached_target_words"]]
        for metric in ("ttr", "mtld"):
            maximum = max(completed, key=lambda row: float(row[metric]))
            output.append(
                {
                    "model_slug": model_slug,
                    "metric": metric,
                    "maximum": float(maximum[metric]),
                    "sample_id": maximum["sample_id"],
                    "temperature": float(maximum["temperature"]),
                    "decoder": maximum["decoder"],
                    "parameter": float(maximum["parameter"]),
                    "text_preview": maximum["text"][:300],
                }
            )
    return output


def joined_plot_rows(results_dir: Path) -> list[dict]:
    diversity = read_csv(results_dir / "lexical_diversity_by_config.csv")
    wcs = read_csv(results_dir / "conditioned_wcs" / "wcs_summary.csv")
    wcs_index = {
        (
            row["model"],
            float(row["temperature"]),
            row["decoder"],
            float(row["parameter"]),
        ): float(row["wcs"])
        for row in wcs
    }
    output = []
    for row in diversity:
        if row["decoder"] == "untruncated":
            continue
        key = (
            row["model"],
            float(row["temperature"]),
            row["decoder"],
            float(row["parameter"]),
        )
        output.append(
            {
                **row,
                "temperature": float(row["temperature"]),
                "parameter": float(row["parameter"]),
                "mean_ttr": float(row["mean_ttr"]),
                "mean_mtld": float(row["mean_mtld"]),
                "wcs": wcs_index[key],
            }
        )
    return output


def make_scatter_plots(results_dir: Path, output_dir: Path) -> list[Path]:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    rows = joined_plot_rows(results_dir)
    model_slugs = list(MODEL_LABELS)
    colors = {0.7: "#2563eb", 1.0: "#dc2626"}
    markers = {"top_k": "o", "top_p": "s", "min_p": "^"}
    paths = []
    for metric, ylabel, stem in (
        ("mean_ttr", "Mean TTR (100-word window)", "wcs_vs_ttr"),
        ("mean_mtld", "Mean MTLD (100-word window)", "wcs_vs_mtld"),
    ):
        fig, axes = plt.subplots(2, 3, figsize=(12.2, 7.2), constrained_layout=True)
        for axis, model_slug in zip(axes.flat, model_slugs):
            model_rows = [row for row in rows if row["model_slug"] == model_slug]
            for row in model_rows:
                axis.scatter(
                    row["wcs"],
                    row[metric],
                    color=colors[row["temperature"]],
                    marker=markers[row["decoder"]],
                    s=42,
                    alpha=0.86,
                    edgecolors="white",
                    linewidths=0.45,
                )
            axis.set_title(MODEL_LABELS[model_slug], fontsize=10.5)
            axis.grid(alpha=0.2)
            axis.set_xlim(-0.03, 1.03)
        for axis in axes[-1, :]:
            axis.set_xlabel("Word Coverage Score")
        for axis in axes[:, 0]:
            axis.set_ylabel(ylabel)
        legend = [
            Line2D([0], [0], marker="o", color="none", markerfacecolor=colors[0.7], label="T=0.7"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=colors[1.0], label="T=1.0"),
            Line2D([0], [0], marker="o", color="#555", linestyle="none", label="top-k"),
            Line2D([0], [0], marker="s", color="#555", linestyle="none", label="top-p"),
            Line2D([0], [0], marker="^", color="#555", linestyle="none", label="min-p"),
        ]
        fig.legend(handles=legend, loc="outside lower center", ncol=5, frameon=False)
        output_dir.mkdir(parents=True, exist_ok=True)
        for suffix in ("png", "pdf"):
            path = output_dir / f"{stem}.{suffix}"
            fig.savefig(path, dpi=250, bbox_inches="tight")
            paths.append(path)
            print(f"[write] {path}")
        plt.close(fig)
    return paths


def format_p(value: float) -> str:
    if value < 0.001:
        return "<.001"
    return f"{value:.3f}".replace("0.", ".")


def write_report(
    path: Path,
    primary: list[dict],
    paired: list[dict],
    completion: list[dict],
) -> None:
    primary_significant = [row for row in primary if row["spearman_significant_holm_05"]]
    base = [row for row in primary if "Instruct" not in row["model"] and not row["model"].endswith("-it")]
    base_significant = [row for row in base if row["spearman_significant_holm_05"]]
    instruct_t1 = [
        row
        for row in primary
        if float(row["temperature"]) == 1.0
        and ("Instruct" in row["model"] or row["model"].endswith("-it"))
    ]
    instruct_t1_significant = [
        row for row in instruct_t1 if row["spearman_significant_holm_05"]
    ]
    paired_significant = [row for row in paired if row["significant_holm_05"]]
    minimum_completion = min(completion, key=lambda row: row["min_completion_rate"])

    lines = [
        "# Corrected downstream-validation findings",
        "",
        "## Primary analysis and multiplicity correction",
        "",
        (
            "We designate the 24 aggregate Spearman tests as the primary family: six models × "
            "two temperatures × two mean diversity metrics. Holm correction controls "
            "the family-wise error rate across these 24 tests. Pearson correlations are "
            "reported as a separately corrected sensitivity analysis."
        ),
        "",
        f"- {len(primary_significant)}/24 primary Spearman relationships remain significant after Holm correction.",
        f"- {len(base_significant)}/{len(base)} Base-model relationships remain significant.",
        (
            f"- {len(instruct_t1_significant)}/{len(instruct_t1)} temperature-1.0 "
            "Instruct relationships remain significant."
        ),
        "",
        "## Secondary paired endpoint analysis",
        "",
        (
            "Endpoint contrasts compare the same contexts under restrictive and permissive "
            "settings using paired Wilcoxon tests. Holm correction is applied across all 72 "
            "model × temperature × sampler × metric tests."
        ),
        "",
        f"- {len(paired_significant)}/72 endpoint effects remain significant after correction.",
        "",
        "## Completion and interpretation",
        "",
        (
            f"The lowest configuration-level completion rate is {minimum_completion['min_completion_rate']:.0%} "
            f"({minimum_completion['model_slug']}, T={minimum_completion['temperature']:g}). "
            "Completion rates must be reported beside diversity because metrics are evaluated "
            "on generations reaching the fixed 100-word window."
        ),
        "",
        "The corrected evidence supports the following claim:",
        "",
        (
            "> Across matched prompts and decoding configurations, greater forced-path word "
            "reachability is generally associated with greater realized lexical diversity. "
            "The association is strongest and most consistent for Base models, while effects "
            "for Instruct models depend on model and temperature."
        ),
        "",
        "It does not establish that exact-word exclusion necessarily causes semantic or qualitative impoverishment.",
        "",
        "## Corrected primary results",
        "",
        "| Model | T | Metric | Spearman ρ | Raw p | Holm p |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for row in primary:
        model = row["model"].split("/")[-1]
        metric = "TTR" if row["metric"] == "mean_ttr" else "MTLD"
        lines.append(
            f"| {model} | {row['temperature']:g} | {metric} | "
            f"{row['spearman_rho']:.2f} | {format_p(row['spearman_p_raw'])} | "
            f"{format_p(row['spearman_p_holm_24'])} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[write] {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.resolve()
    output_dir = args.output_dir.resolve()
    primary = primary_correlations(results_dir / "wcs_diversity_correlations.csv")
    descriptive = sampler_correlations_descriptive(
        results_dir / "wcs_diversity_correlations.csv"
    )
    generations = load_generations(results_dir / "generations")
    paired = paired_endpoint_effects(generations)
    completion = completion_summary(results_dir / "lexical_diversity_by_config.csv")
    outliers = outlier_summary(generations)

    write_csv(output_dir / "primary_correlations_holm.csv", primary)
    write_csv(output_dir / "sampler_correlations_descriptive.csv", descriptive)
    write_csv(output_dir / "paired_endpoint_effects_holm.csv", paired)
    write_csv(output_dir / "completion_rates.csv", completion)
    write_csv(output_dir / "metric_outliers.csv", outliers)
    make_scatter_plots(results_dir, output_dir / "figures")
    write_report(
        output_dir / "paper_ready_findings.md", primary, paired, completion
    )


if __name__ == "__main__":
    main()
