#!/usr/bin/env python
"""Build ranked word-frequency TSV files from local text, CSV, JSON, or lyric trees."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Iterator

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from wcs.dataset_builder import WORD_RE, normalize_target_word


GUTENBERG_START_RE = re.compile(r"\*\*\* START OF (?:THE )?PROJECT GUTENBERG EBOOK.*?\*\*\*", re.I | re.S)
GUTENBERG_END_RE = re.compile(r"\*\*\* END OF (?:THE )?PROJECT GUTENBERG EBOOK.*", re.I | re.S)


def strip_gutenberg_boilerplate(text: str) -> str:
    start = GUTENBERG_START_RE.search(text)
    if start:
        text = text[start.end() :]
    end = GUTENBERG_END_RE.search(text)
    if end:
        text = text[: end.start()]
    return text


def iter_text_from_json(value: object) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from iter_text_from_json(item)
    elif isinstance(value, dict):
        for key, item in value.items():
            if key.lower() in {"url", "album_url"}:
                continue
            yield from iter_text_from_json(item)


def read_csv_text(path: Path) -> str:
    with path.open("r", encoding="utf-8-sig", errors="ignore", newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t")
        reader = csv.DictReader(handle, dialect=dialect)
        if not reader.fieldnames:
            return ""
        lyric_field = next((name for name in reader.fieldnames if name.lower() == "lyrics"), None)
        fields = [lyric_field] if lyric_field else reader.fieldnames
        return "\n".join(
            row.get(field, "")
            for row in reader
            for field in fields
            if field and row.get(field)
        )


def read_text_payload(path: Path, strip_gutenberg: bool) -> str:
    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8-sig", errors="ignore"))
        text = "\n".join(iter_text_from_json(payload))
    elif suffix == ".csv":
        text = read_csv_text(path)
    else:
        text = path.read_text(encoding="utf-8-sig", errors="ignore")
    return strip_gutenberg_boilerplate(text) if strip_gutenberg else text


def count_words(texts: Iterable[str], min_word_length: int) -> Counter[str]:
    counts: Counter[str] = Counter()
    for text in texts:
        for match in WORD_RE.finditer(text):
            word = normalize_target_word(match.group(0))
            if word and len(word) >= min_word_length:
                counts[word] += 1
    return counts


def write_frequency(path: Path, counts: Counter[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        for rank, (word, count) in enumerate(rows, start=1):
            writer.writerow([rank, word, count])


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", value.strip().lower()).strip("_")
    return slug or "words"


def discover_grouped_inputs(paths: Iterable[Path]) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = defaultdict(list)
    for raw_path in paths:
        path = raw_path.resolve()
        if path.is_file():
            grouped[path.stem].append(path)
            continue
        if not path.is_dir():
            raise FileNotFoundError(path)
        for child in sorted(path.iterdir()):
            if child.name.startswith("."):
                continue
            if child.is_file() and child.suffix.lower() in {".txt", ".text", ".csv", ".json"}:
                grouped[child.stem].append(child)
            elif child.is_dir():
                files = [
                    item
                    for item in sorted(child.rglob("*"))
                    if item.is_file() and item.suffix.lower() in {".txt", ".text", ".csv", ".json"}
                ]
                if files:
                    grouped[child.name].extend(files)
    return dict(grouped)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create ranked word-frequency TSV files.")
    parser.add_argument("inputs", type=Path, nargs="+", help="Files or directories to scan.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/wordlists"))
    parser.add_argument("--combined-name", default="combined")
    parser.add_argument("--group-name", default=None, help="Force all inputs into one named group.")
    parser.add_argument("--min-word-length", type=int, default=3)
    parser.add_argument("--strip-gutenberg", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.group_name:
        grouped = {args.group_name: [path.resolve() for path in args.inputs]}
    else:
        grouped = discover_grouped_inputs(args.inputs)

    combined: Counter[str] = Counter()
    for group, paths in sorted(grouped.items()):
        texts = (read_text_payload(path, args.strip_gutenberg) for path in paths)
        counts = count_words(texts, args.min_word_length)
        combined.update(counts)
        output = args.output_dir / f"{slugify(group)}.tsv"
        write_frequency(output, counts)
        print(f"Wrote {len(counts)} words from {len(paths)} files to {output}")

    if len(grouped) > 1:
        output = args.output_dir / f"{slugify(args.combined_name)}.tsv"
        write_frequency(output, combined)
        print(f"Wrote {len(combined)} combined words to {output}")


if __name__ == "__main__":
    main()
