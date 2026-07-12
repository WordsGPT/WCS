#!/usr/bin/env python
"""Estimate pooled WCS/diversity effects with repeated-measures controls."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from wcs.metrics import (
    word_survives_min_p,
    word_survives_top_k,
    word_survives_top_p,
)


def audit_index(audit_dir: Path) -> dict[tuple[str, float, str], list[dict]]:
    grouped: dict[tuple[str, float, str], list[dict]] = defaultdict(list)
    for path in sorted(audit_dir.glob("audit.*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                grouped[(row["model"], float(row["temperature"]), row["sample_id"])].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: int(row["word_token_index"]))
    return grouped


def survives(rows: list[dict], decoder: str, parameter: float) -> bool:
    if decoder == "top_k":
        return word_survives_top_k(rows, int(parameter))
    if decoder == "top_p":
        return word_survives_top_p(rows, parameter)
    if decoder == "min_p":
        return word_survives_min_p(rows, parameter)
    raise ValueError(f"Unsupported decoder: {decoder}")


def joined_rows(results_dir: Path) -> list[dict]:
    audits = audit_index(results_dir / "conditioned_wcs")
    output = []
    for path in sorted((results_dir / "generations").glob("generation.*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row["decoder"] == "untruncated" or not row["reached_target_words"]:
                    continue
                key = (row["model"], float(row["temperature"]), row["sample_id"])
                token_rows = audits.get(key)
                if token_rows is None:
                    raise KeyError(f"No forced-path audit for {key}")
                parameter = float(row["parameter"])
                output.append(
                    {
                        "sample_id": row["sample_id"],
                        "configuration": "|".join(
                            (
                                row["model"],
                                f"{float(row['temperature']):g}",
                                row["decoder"],
                                f"{parameter:g}",
                            )
                        ),
                        "survives": float(survives(token_rows, row["decoder"], parameter)),
                        "ttr": float(row["ttr"]),
                        "mtld": float(row["mtld"]),
                    }
                )
    return output


def residualize(values: np.ndarray, group_a: np.ndarray, group_b: np.ndarray) -> np.ndarray:
    """Remove two additive fixed effects by alternating projections."""
    residual = values.astype(float) - values.mean()
    for _ in range(10_000):
        previous = residual.copy()
        for groups in (group_a, group_b):
            totals = np.bincount(groups, weights=residual)
            counts = np.bincount(groups)
            residual -= (totals / counts)[groups]
        if np.max(np.abs(residual - previous)) < 1e-12:
            break
    return residual


def pooled_effect(rows: list[dict], metric: str) -> dict:
    samples = {value: index for index, value in enumerate(sorted({row["sample_id"] for row in rows}))}
    configs = {value: index for index, value in enumerate(sorted({row["configuration"] for row in rows}))}
    sample_ids = np.asarray([samples[row["sample_id"]] for row in rows], dtype=int)
    config_ids = np.asarray([configs[row["configuration"]] for row in rows], dtype=int)
    x = np.asarray([row["survives"] for row in rows], dtype=float)
    y = np.asarray([row[metric] for row in rows], dtype=float)

    x_residual = residualize(x, sample_ids, config_ids)
    y_residual = residualize(y, sample_ids, config_ids)
    denominator = float(x_residual @ x_residual)
    estimate = float(x_residual @ y_residual / denominator)
    errors = y_residual - estimate * x_residual

    # One-way cluster-robust sandwich variance over the shared FineWeb contexts.
    scores = np.bincount(sample_ids, weights=x_residual * errors)
    clusters = len(samples)
    observations = len(rows)
    fixed_effect_parameters = len(samples) + len(configs) - 1
    correction = (clusters / (clusters - 1)) * (
        (observations - 1) / (observations - fixed_effect_parameters - 1)
    )
    variance = correction * float(scores @ scores) / (denominator * denominator)
    standard_error = math.sqrt(max(variance, 0.0))
    z_value = estimate / standard_error
    p_value = math.erfc(abs(z_value) / math.sqrt(2.0))
    return {
        "metric": metric,
        "effect_reachable_minus_unreachable": estimate,
        "cluster_robust_se": standard_error,
        "ci95_low": estimate - 1.96 * standard_error,
        "ci95_high": estimate + 1.96 * standard_error,
        "z_value": z_value,
        "p_value": p_value,
        "observations": observations,
        "contexts": clusters,
        "configurations": len(configs),
        "reachable_observations": int(x.sum()),
        "unreachable_observations": int(observations - x.sum()),
        "fixed_effects": "context and model×temperature×decoder×parameter",
        "uncertainty": "sandwich standard errors clustered by context",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = joined_rows(args.results_dir)
    results = [pooled_effect(rows, metric) for metric in ("ttr", "mtld")]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
