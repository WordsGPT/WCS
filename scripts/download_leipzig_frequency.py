#!/usr/bin/env python
"""Download a Leipzig Corpora Collection word list and convert it to WCS TSV."""

from __future__ import annotations

import argparse
import csv
import tarfile
import urllib.request
from collections import Counter
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from wcs.dataset_builder import normalize_target_word


DEFAULT_CORPUS = "spa_news_2023_1M"
DEFAULT_BASE_URL = "https://downloads.wortschatz-leipzig.de/corpora"


def download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as response:
        with path.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)


def find_words_member(archive_path: Path) -> str:
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            if member.isfile() and (
                member.name.endswith("_words.txt") or member.name.endswith("-words.txt")
            ):
                return member.name
    raise FileNotFoundError(f"No *_words.txt file found in {archive_path}")


def load_counts_from_archive(archive_path: Path, min_word_length: int) -> Counter[str]:
    words_member = find_words_member(archive_path)
    counts: Counter[str] = Counter()
    with tarfile.open(archive_path, "r:gz") as archive:
        extracted = archive.extractfile(words_member)
        if extracted is None:
            raise FileNotFoundError(words_member)
        for raw_line in extracted:
            line = raw_line.decode("utf-8", errors="ignore").strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            word = normalize_target_word(parts[1])
            if not word or len(word) < min_word_length:
                continue
            try:
                count = int(parts[2])
            except ValueError:
                continue
            counts[word] += count
    return counts


def write_frequency(path: Path, counts: Counter[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        for rank, (word, count) in enumerate(rows, start=1):
            writer.writerow([rank, word, count])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and convert a Leipzig Corpora Collection frequency list."
    )
    parser.add_argument("--corpus", default=DEFAULT_CORPUS)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--archive", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("data/raw/spanish_frequency.tsv"))
    parser.add_argument("--min-word-length", type=int, default=1)
    parser.add_argument("--force-download", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    archive_path = args.archive or Path("data/raw/leipzig") / f"{args.corpus}.tar.gz"
    url = f"{args.base_url.rstrip('/')}/{args.corpus}.tar.gz"

    if args.force_download or not archive_path.exists():
        print(f"Downloading {url} to {archive_path}")
        download(url, archive_path)
    else:
        print(f"Using existing archive {archive_path}")

    counts = load_counts_from_archive(archive_path, args.min_word_length)
    write_frequency(args.output, counts)
    print(f"Wrote {len(counts)} words to {args.output}")


if __name__ == "__main__":
    main()
