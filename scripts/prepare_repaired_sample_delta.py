#!/usr/bin/env python
"""Extract only samples whose evaluated context changed after dataset repair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


EVALUATED_FIELDS = (
    "word",
    "prefix",
    "matched_text",
    "source_path",
    "match_start_char",
    "match_end_char",
)


def load_rows(path: Path) -> tuple[list[str], dict[str, dict]]:
    order: list[str] = []
    rows: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = str(row["id"])
            if sample_id in rows:
                raise ValueError(f"Duplicate sample ID {sample_id!r} in {path}")
            order.append(sample_id)
            rows[sample_id] = row
    return order, rows


def changed_sample_ids(original: dict[str, dict], repaired: dict[str, dict]) -> set[str]:
    if original.keys() != repaired.keys():
        missing = sorted(original.keys() - repaired.keys())
        added = sorted(repaired.keys() - original.keys())
        raise ValueError(
            f"Sample ID sets differ; missing={missing[:5]}, added={added[:5]}"
        )
    return {
        sample_id
        for sample_id in original
        if any(
            original[sample_id].get(field) != repaired[sample_id].get(field)
            for field in EVALUATED_FIELDS
        )
    }


def write_delta(original_path: Path, repaired_path: Path, output_path: Path) -> int:
    _original_order, original = load_rows(original_path)
    repaired_order, repaired = load_rows(repaired_path)
    changed = changed_sample_ids(original, repaired)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for sample_id in repaired_order:
            if sample_id in changed:
                handle.write(json.dumps(repaired[sample_id], ensure_ascii=False) + "\n")
    temporary.replace(output_path)
    return len(changed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write only repaired samples whose model input changed."
    )
    parser.add_argument(
        "--original",
        type=Path,
        default=Path("data/processed/samples.spanish_pd_books.100x10.jsonl"),
    )
    parser.add_argument(
        "--repaired",
        type=Path,
        default=Path(
            "data/processed/samples.spanish_pd_books.100x10.repaired.jsonl"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/processed/samples.spanish_pd_books.repaired_delta.jsonl"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = write_delta(args.original, args.repaired, args.output)
    print(f"Wrote {count} changed samples to {args.output}")


if __name__ == "__main__":
    main()
