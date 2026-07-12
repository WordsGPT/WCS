#!/usr/bin/env python
"""Repeated-measures Spearman correlations over decoder configurations."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import rankdata


def repeated_spearman(groups: dict[tuple[str, float], list[dict]], models: list[str], metric: str) -> float:
    x_values: list[float] = []
    y_values: list[float] = []
    for model in models:
        for (group_model, _temperature), rows in groups.items():
            if group_model != model:
                continue
            x_ranks = rankdata([float(row["wcs"]) for row in rows])
            y_ranks = rankdata([float(row[metric]) for row in rows])
            x_values.extend(x_ranks - x_ranks.mean())
            y_values.extend(y_ranks - y_ranks.mean())
    return float(np.corrcoef(x_values, y_values)[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=10298)
    args = parser.parse_args()

    value = json.loads(args.data.read_text(encoding="utf-8"))
    groups: dict[tuple[str, float], list[dict]] = defaultdict(list)
    for row in value["rows"]:
        groups[(row["model"], float(row["temperature"]))].append(row)
    models = sorted({model for model, _temperature in groups})
    rng = np.random.default_rng(args.seed)
    output = []
    for metric in ("mean_ttr", "mean_mtld"):
        estimate = repeated_spearman(groups, models, metric)
        bootstrap = np.asarray(
            [
                repeated_spearman(
                    groups,
                    list(rng.choice(models, size=len(models), replace=True)),
                    metric,
                )
                for _ in range(args.bootstrap_draws)
            ]
        )
        low, high = np.quantile(bootstrap, (0.025, 0.975))
        output.append(
            {
                "metric": metric,
                "repeated_spearman_rho": estimate,
                "ci95_low": float(low),
                "ci95_high": float(high),
                "configuration_averages": sum(len(rows) for rows in groups.values()),
                "model_temperature_panels": len(groups),
                "models": len(models),
                "configurations_per_panel": min(len(rows) for rows in groups.values()),
                "ci_method": f"model-cluster bootstrap, {args.bootstrap_draws} draws",
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
