"""Aggregate forced-path audit rows into WCS summaries."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator


DEFAULT_TOP_K = tuple(range(1, 21)) + tuple(range(25, 81, 5))
DEFAULT_TOP_P = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99)
DEFAULT_MIN_P = (0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10)


@dataclass(frozen=True)
class WcsSummaryRow:
    temperature: float
    model: str
    decoder: str
    parameter: float
    wcs: float
    covered_words: int
    total_words: int
    audit_path: str


@dataclass(frozen=True)
class WcsWordSummaryRow:
    temperature: float
    model: str
    decoder: str
    parameter: float
    word_any_wcs: float
    word_all_wcs: float
    covered_words_any: int
    covered_words_all: int
    total_words: int
    covered_contexts: int
    total_contexts: int
    audit_path: str


def parse_number_list(raw: str, number_type: type = float) -> list:
    values = []
    for part in raw.split(","):
        stripped = part.strip()
        if stripped:
            values.append(number_type(stripped))
    return values


def iter_audit_rows(paths: Iterable[Path]) -> Iterator[dict]:
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    row["_audit_path"] = str(path)
                    yield row


def group_rows_by_word(rows: Iterable[dict]) -> dict[tuple[str, float, str, str], list[dict]]:
    grouped: dict[tuple[str, float, str, str], list[dict]] = {}
    for row in rows:
        key = (row["model"], float(row.get("temperature", 1.0)), row["sample_id"], row["_audit_path"])
        grouped.setdefault(key, []).append(row)
    return grouped


def group_rows_by_target_word(rows: Iterable[dict]) -> dict[tuple[str, float, str, str], dict[str, list[dict]]]:
    grouped: dict[tuple[str, float, str, str], dict[str, list[dict]]] = {}
    for row in rows:
        key = (row["model"], float(row.get("temperature", 1.0)), row["word"], row["_audit_path"])
        grouped.setdefault(key, {}).setdefault(row["sample_id"], []).append(row)
    return grouped


def word_survives_top_k(rows: list[dict], k: int) -> bool:
    return all(int(row["rank"]) <= k for row in rows)


def word_survives_top_p(rows: list[dict], p: float) -> bool:
    return all(float(row["cumulative_probability"]) <= p for row in rows)


def word_survives_min_p(rows: list[dict], min_p: float) -> bool:
    return all(float(row["probability_ratio_to_top"]) >= min_p for row in rows)


def summarize_wcs(
    audit_paths: Iterable[Path],
    top_k_values: Iterable[int] = DEFAULT_TOP_K,
    top_p_values: Iterable[float] = DEFAULT_TOP_P,
    min_p_values: Iterable[float] = DEFAULT_MIN_P,
) -> list[WcsSummaryRow]:
    grouped = group_rows_by_word(iter_audit_rows(audit_paths))
    by_model_path: dict[tuple[str, float, str], list[list[dict]]] = {}
    for (model, temperature, _sample_id, audit_path), rows in grouped.items():
        rows = sorted(rows, key=lambda row: int(row["word_token_index"]))
        by_model_path.setdefault((model, temperature, audit_path), []).append(rows)

    summaries: list[WcsSummaryRow] = []
    for (model, temperature, audit_path), word_rows in sorted(by_model_path.items()):
        total_words = len(word_rows)
        for k in top_k_values:
            covered = sum(1 for rows in word_rows if word_survives_top_k(rows, int(k)))
            summaries.append(
                WcsSummaryRow(
                    temperature=temperature,
                    model=model,
                    decoder="top_k",
                    parameter=float(k),
                    wcs=covered / total_words if total_words else 0.0,
                    covered_words=covered,
                    total_words=total_words,
                    audit_path=audit_path,
                )
            )
        for p in top_p_values:
            covered = sum(1 for rows in word_rows if word_survives_top_p(rows, float(p)))
            summaries.append(
                WcsSummaryRow(
                    temperature=temperature,
                    model=model,
                    decoder="top_p",
                    parameter=float(p),
                    wcs=covered / total_words if total_words else 0.0,
                    covered_words=covered,
                    total_words=total_words,
                    audit_path=audit_path,
                )
            )
        for min_p in min_p_values:
            covered = sum(1 for rows in word_rows if word_survives_min_p(rows, float(min_p)))
            summaries.append(
                WcsSummaryRow(
                    temperature=temperature,
                    model=model,
                    decoder="min_p",
                    parameter=float(min_p),
                    wcs=covered / total_words if total_words else 0.0,
                    covered_words=covered,
                    total_words=total_words,
                    audit_path=audit_path,
                )
            )
    return summaries


def summarize_wcs_by_target_word(
    audit_paths: Iterable[Path],
    top_k_values: Iterable[int] = DEFAULT_TOP_K,
    top_p_values: Iterable[float] = DEFAULT_TOP_P,
    min_p_values: Iterable[float] = DEFAULT_MIN_P,
) -> list[WcsWordSummaryRow]:
    grouped = group_rows_by_target_word(iter_audit_rows(audit_paths))
    by_model_path: dict[tuple[str, float, str], list[list[list[dict]]]] = {}
    for (model, temperature, _word, audit_path), sample_groups in grouped.items():
        context_rows = [
            sorted(rows, key=lambda row: int(row["word_token_index"]))
            for _sample_id, rows in sorted(sample_groups.items())
        ]
        by_model_path.setdefault((model, temperature, audit_path), []).append(context_rows)

    summaries: list[WcsWordSummaryRow] = []
    for (model, temperature, audit_path), word_contexts in sorted(by_model_path.items()):
        total_words = len(word_contexts)
        total_contexts = sum(len(contexts) for contexts in word_contexts)

        def append_summary(decoder: str, parameter: float, context_survival: list[list[bool]]) -> None:
            covered_words_any = sum(1 for contexts in context_survival if any(contexts))
            covered_words_all = sum(1 for contexts in context_survival if contexts and all(contexts))
            covered_contexts = sum(1 for contexts in context_survival for survives in contexts if survives)
            summaries.append(
                WcsWordSummaryRow(
                    temperature=temperature,
                    model=model,
                    decoder=decoder,
                    parameter=float(parameter),
                    word_any_wcs=covered_words_any / total_words if total_words else 0.0,
                    word_all_wcs=covered_words_all / total_words if total_words else 0.0,
                    covered_words_any=covered_words_any,
                    covered_words_all=covered_words_all,
                    total_words=total_words,
                    covered_contexts=covered_contexts,
                    total_contexts=total_contexts,
                    audit_path=audit_path,
                )
            )

        for k in top_k_values:
            append_summary(
                "top_k",
                float(k),
                [
                    [word_survives_top_k(rows, int(k)) for rows in contexts]
                    for contexts in word_contexts
                ],
            )
        for p in top_p_values:
            append_summary(
                "top_p",
                float(p),
                [
                    [word_survives_top_p(rows, float(p)) for rows in contexts]
                    for contexts in word_contexts
                ],
            )
        for min_p in min_p_values:
            append_summary(
                "min_p",
                float(min_p),
                [
                    [word_survives_min_p(rows, float(min_p)) for rows in contexts]
                    for contexts in word_contexts
                ],
            )
    return summaries


def write_summary_csv(rows: Iterable[WcsSummaryRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "temperature",
        "model",
        "decoder",
        "parameter",
        "wcs",
        "covered_words",
        "total_words",
        "audit_path",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def write_word_summary_csv(rows: Iterable[WcsWordSummaryRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "temperature",
        "model",
        "decoder",
        "parameter",
        "word_any_wcs",
        "word_all_wcs",
        "covered_words_any",
        "covered_words_all",
        "total_words",
        "covered_contexts",
        "total_contexts",
        "audit_path",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def read_summary_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def plot_summary(summary_csv: Path, output_dir: Path) -> list[Path]:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for plotting") from exc

    rows = read_summary_csv(summary_csv)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []

    for decoder, filename in (
        ("top_k", "wcs_vs_topk.png"),
        ("top_p", "wcs_vs_topp.png"),
        ("min_p", "wcs_vs_minp.png"),
    ):
        decoder_rows = [row for row in rows if row["decoder"] == decoder]
        if not decoder_rows:
            continue
        models = sorted({row["model"] for row in decoder_rows})
        plt.figure(figsize=(8, 5))
        for model in models:
            model_rows = [row for row in decoder_rows if row["model"] == model]
            model_rows.sort(key=lambda row: float(row["parameter"]))
            x = [float(row["parameter"]) for row in model_rows]
            y = [float(row["wcs"]) for row in model_rows]
            plt.plot(x, y, marker="o", linewidth=1.5, label=model)
        plt.xlabel(decoder.replace("_", "-"))
        plt.ylabel("WCS")
        plt.ylim(0, 1.02)
        plt.grid(True, alpha=0.25)
        plt.legend(fontsize="small")
        plt.tight_layout()
        output_path = output_dir / filename
        plt.savefig(output_path, dpi=200)
        plt.close()
        output_paths.append(output_path)
    return output_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate WCS audit JSONL files.")
    parser.add_argument("--audits", type=Path, nargs="+", required=True)
    parser.add_argument("--summary", type=Path, default=Path("results/wcs_summary.csv"))
    parser.add_argument("--word-summary", type=Path, default=None)
    parser.add_argument("--plot-dir", type=Path, default=None)
    parser.add_argument("--top-k", default=",".join(str(v) for v in DEFAULT_TOP_K))
    parser.add_argument("--top-p", default=",".join(str(v) for v in DEFAULT_TOP_P))
    parser.add_argument("--min-p", default=",".join(str(v) for v in DEFAULT_MIN_P))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    top_k_values = parse_number_list(args.top_k, int)
    top_p_values = parse_number_list(args.top_p, float)
    min_p_values = parse_number_list(args.min_p, float)

    summaries = summarize_wcs(
        audit_paths=args.audits,
        top_k_values=top_k_values,
        top_p_values=top_p_values,
        min_p_values=min_p_values,
    )
    write_summary_csv(summaries, args.summary)
    print(f"Wrote WCS summary to {args.summary}")

    if args.word_summary is not None:
        word_summaries = summarize_wcs_by_target_word(
            audit_paths=args.audits,
            top_k_values=top_k_values,
            top_p_values=top_p_values,
            min_p_values=min_p_values,
        )
        write_word_summary_csv(word_summaries, args.word_summary)
        print(f"Wrote word-level WCS summary to {args.word_summary}")

    if args.plot_dir is not None:
        output_paths = plot_summary(args.summary, args.plot_dir)
        for output_path in output_paths:
            print(f"Wrote plot to {output_path}")


if __name__ == "__main__":
    main()
