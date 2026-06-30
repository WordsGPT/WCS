#!/usr/bin/env python
"""Summarize Gemini context decisions and show rejected examples by reason."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_LOG = Path("data/processed/spanish_pd_books.coherence.jsonl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--examples-per-reason", type=int, default=2)
    parser.add_argument("--excerpt-chars", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    counts: Counter[str] = Counter()
    examples: dict[str, list[dict[str, object]]] = defaultdict(list)
    total = 0
    accepted = 0
    with args.log.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            total += 1
            if row.get("accepted") is True:
                accepted += 1
                reason = "accepted"
            else:
                reason = str(row.get("reason", "unknown"))
                if len(examples[reason]) < args.examples_per_reason:
                    examples[reason].append(row)
            counts[reason] += 1

    print(f"Decisions: {total}")
    if total:
        print(f"Accepted: {accepted} ({accepted / total:.1%})")
    print("Reasons:")
    for reason, count in counts.most_common():
        print(f"  {reason}: {count} ({count / total:.1%})")

    for reason, rows in sorted(examples.items()):
        print(f"\n[{reason}]")
        for row in rows:
            excerpt = " ".join(str(row.get("excerpt", "")).split())
            if len(excerpt) > args.excerpt_chars:
                excerpt = "…" + excerpt[-args.excerpt_chars :]
            print(
                f"- word={row.get('word')!r} rank={row.get('rank')} "
                f"source={row.get('source_path')}"
            )
            print(f"  Gemini: {row.get('note')}")
            print(f"  Excerpt: {excerpt}")


if __name__ == "__main__":
    main()
