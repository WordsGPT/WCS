#!/usr/bin/env python
"""CLI wrapper for word-level WCS aggregation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from wcs.metrics import (
    DEFAULT_MIN_P,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    parse_number_list,
    summarize_wcs_by_target_word,
    write_word_summary_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate WCS audit JSONL files by unique target word. "
            "A word is counted in word_any_wcs if at least one context survives."
        )
    )
    parser.add_argument("--audits", type=Path, nargs="+", required=True)
    parser.add_argument("--summary", type=Path, default=Path("results/wcs_word_summary.csv"))
    parser.add_argument("--top-k", default=",".join(str(v) for v in DEFAULT_TOP_K))
    parser.add_argument("--top-p", default=",".join(str(v) for v in DEFAULT_TOP_P))
    parser.add_argument("--min-p", default=",".join(str(v) for v in DEFAULT_MIN_P))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries = summarize_wcs_by_target_word(
        audit_paths=args.audits,
        top_k_values=parse_number_list(args.top_k, int),
        top_p_values=parse_number_list(args.top_p, float),
        min_p_values=parse_number_list(args.min_p, float),
    )
    write_word_summary_csv(summaries, args.summary)
    print(f"Wrote word-level WCS summary to {args.summary}")


if __name__ == "__main__":
    main()
