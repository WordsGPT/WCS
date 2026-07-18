#!/usr/bin/env python
"""Summarize WCS under intersections of top-k and top-p filters."""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from wcs.metrics import (
    group_rows_by_target_word,
    group_rows_by_word,
    iter_audit_rows,
)


@dataclass(frozen=True)
class Recipe:
    name: str
    top_k: int
    top_p: float


def parse_recipe(raw: str) -> Recipe:
    name, separator, values = raw.partition(":")
    if not separator:
        raise ValueError("recipes must use NAME:TOP_K:TOP_P")
    top_k, separator, top_p = values.partition(":")
    if not separator or not name or not top_k or not top_p:
        raise ValueError("recipes must use NAME:TOP_K:TOP_P")
    recipe = Recipe(name=name, top_k=int(top_k), top_p=float(top_p))
    if recipe.top_k <= 0:
        raise ValueError("recipe TOP_K must be positive")
    if not 0 < recipe.top_p <= 1:
        raise ValueError("recipe TOP_P must be in (0, 1]")
    return recipe


def survives(rows: list[dict], recipe: Recipe) -> bool:
    """Match Transformers' sequential top-k then top-p warpers.

    Top-p is evaluated after top-k masks and renormalizes the distribution, so
    this is not merely the intersection of independently computed top-k and
    top-p support sets.
    """

    for row in rows:
        if int(row["rank"]) > recipe.top_k:
            return False
        saved_probabilities = row.get("top_5_probs") or []
        if len(saved_probabilities) < recipe.top_k:
            raise ValueError(
                f"audit stores {len(saved_probabilities)} top probabilities, "
                f"but recipe {recipe.name!r} requires {recipe.top_k}"
            )
        retained_mass = sum(
            float(probability)
            for probability in saved_probabilities[: recipe.top_k]
        )
        mass_strictly_above = (
            float(row["cumulative_probability"]) - float(row["probability"])
        )
        if retained_mass <= 0 or mass_strictly_above / retained_mass > recipe.top_p:
            return False
    return True


def context_rows(audit_paths: list[Path], recipes: list[Recipe]) -> list[dict]:
    grouped = group_rows_by_word(iter_audit_rows(audit_paths))
    by_model_path: dict[tuple[str, float, str], list[list[dict]]] = {}
    for (model, temperature, _sample_id, audit_path), rows in grouped.items():
        ordered = sorted(rows, key=lambda row: int(row["word_token_index"]))
        by_model_path.setdefault((model, temperature, audit_path), []).append(ordered)

    output: list[dict] = []
    for (model, temperature, audit_path), words in sorted(by_model_path.items()):
        for recipe in recipes:
            covered = sum(survives(rows, recipe) for rows in words)
            output.append(
                {
                    "temperature": temperature,
                    "model": model,
                    "recipe": recipe.name,
                    "top_k": recipe.top_k,
                    "top_p": recipe.top_p,
                    "wcs": covered / len(words) if words else 0.0,
                    "covered_words": covered,
                    "total_words": len(words),
                    "audit_path": audit_path,
                }
            )
    return output


def target_word_rows(audit_paths: list[Path], recipes: list[Recipe]) -> list[dict]:
    grouped = group_rows_by_target_word(iter_audit_rows(audit_paths))
    by_model_path: dict[tuple[str, float, str], list[list[list[dict]]]] = {}
    for (model, temperature, _word, audit_path), sample_groups in grouped.items():
        contexts = [
            sorted(rows, key=lambda row: int(row["word_token_index"]))
            for _sample_id, rows in sorted(sample_groups.items())
        ]
        by_model_path.setdefault((model, temperature, audit_path), []).append(contexts)

    output: list[dict] = []
    for (model, temperature, audit_path), words in sorted(by_model_path.items()):
        total_contexts = sum(len(contexts) for contexts in words)
        for recipe in recipes:
            survival = [
                [survives(rows, recipe) for rows in contexts] for contexts in words
            ]
            covered_any = sum(any(contexts) for contexts in survival)
            covered_all = sum(bool(contexts) and all(contexts) for contexts in survival)
            covered_contexts = sum(
                value for contexts in survival for value in contexts
            )
            output.append(
                {
                    "temperature": temperature,
                    "model": model,
                    "recipe": recipe.name,
                    "top_k": recipe.top_k,
                    "top_p": recipe.top_p,
                    "word_any_wcs": covered_any / len(words) if words else 0.0,
                    "word_all_wcs": covered_all / len(words) if words else 0.0,
                    "covered_words_any": covered_any,
                    "covered_words_all": covered_all,
                    "total_words": len(words),
                    "covered_contexts": covered_contexts,
                    "total_contexts": total_contexts,
                    "audit_path": audit_path,
                }
            )
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("no audit rows were found")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute WCS for joint top-k and top-p provider recipes."
    )
    parser.add_argument("--audits", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--recipe",
        action="append",
        required=True,
        help="Combined filter in NAME:TOP_K:TOP_P form; may be repeated.",
    )
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--word-summary", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    recipes = [parse_recipe(raw) for raw in args.recipe]
    write_csv(args.summary, context_rows(args.audits, recipes))
    write_csv(args.word_summary, target_word_rows(args.audits, recipes))
    print(f"Wrote combined-filter WCS summary to {args.summary}")
    print(f"Wrote combined-filter word summary to {args.word_summary}")


if __name__ == "__main__":
    main()
