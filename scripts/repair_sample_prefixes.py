#!/usr/bin/env python
"""Repair existing sample prefixes by restoring raw corpus punctuation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from wcs.dataset_builder import WORD_RE


def raw_prefix_for_sample(row: dict, root: Path) -> str:
    source_path = Path(row["source_path"])
    if not source_path.is_absolute():
        source_path = root / source_path
    text = source_path.read_text(encoding="utf-8", errors="ignore")
    match_start = int(row["match_start_char"])
    context_tokens = int(row["context_token_count"])
    token_matches = list(WORD_RE.finditer(text[:match_start]))
    if len(token_matches) < context_tokens:
        raise ValueError(f"Not enough context tokens for {row['id']} in {source_path}")
    start = token_matches[-context_tokens].start()
    return text[start:match_start].strip()


def repair_samples(input_path: Path, output_path: Path, root: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with input_path.open("r", encoding="utf-8") as source, output_path.open(
        "w",
        encoding="utf-8",
    ) as target:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            row["prefix"] = raw_prefix_for_sample(row, root)
            target.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Restore punctuation-preserving prefixes in an existing samples JSONL file."
    )
    parser.add_argument("--input", type=Path, default=Path("data/processed/samples.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/samples.jsonl"))
    parser.add_argument("--backup", type=Path, default=Path("data/processed/samples.no_punctuation.jsonl"))
    parser.add_argument("--root", type=Path, default=Path("."))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input
    output_path = args.output
    if input_path.resolve() == output_path.resolve():
        args.backup.parent.mkdir(parents=True, exist_ok=True)
        args.backup.write_text(input_path.read_text(encoding="utf-8"), encoding="utf-8")
        temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
        count = repair_samples(input_path, temp_path, args.root)
        temp_path.replace(output_path)
        print(f"Backed up original samples to {args.backup}")
    else:
        count = repair_samples(input_path, output_path, args.root)
    print(f"Wrote {count} repaired samples to {output_path}")


if __name__ == "__main__":
    main()
