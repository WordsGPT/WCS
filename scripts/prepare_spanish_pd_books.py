#!/usr/bin/env python
"""Prepare disjoint frequency and context splits from Spanish-PD-Books."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Iterator

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from wcs.dataset_builder import WORD_RE, normalize_target_word


DEFAULT_REPO = "PleIAs/Spanish-PD-Books"
PARQUET_INDEX_RE = re.compile(r"spanish_pd_(\d+)\.parquet$")
TEXT_COLUMNS = ("text", "content", "body")
LANGUAGE_COLUMNS = ("real_lang", "lang", "language")


def require_pyarrow():
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise SystemExit(
            "Spanish-PD-Books preparation requires pyarrow. "
            "Install it with: python -m pip install -e '.[pd-books]'"
        ) from exc
    return parquet


def parquet_sort_key(filename: str) -> tuple[int, str]:
    match = PARQUET_INDEX_RE.search(filename)
    return (int(match.group(1)) if match else sys.maxsize, filename)


def discover_remote_parquets(repo_id: str, revision: str) -> list[str]:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise SystemExit(
            "Install the pd-books dependencies with: "
            "python -m pip install -e '.[pd-books]'"
        ) from exc
    files = HfApi().list_repo_files(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
    )
    return sorted(
        (filename for filename in files if filename.endswith(".parquet")),
        key=parquet_sort_key,
    )


def download_parquets(
    filenames: list[str],
    *,
    repo_id: str,
    revision: str,
    parquet_dir: Path,
) -> list[Path]:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise SystemExit(
            "Install the pd-books dependencies with: "
            "python -m pip install -e '.[pd-books]'"
        ) from exc
    parquet_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for index, filename in enumerate(filenames, start=1):
        print(f"[download {index}/{len(filenames)}] {filename}", flush=True)
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            revision=revision,
            local_dir=parquet_dir,
        )
        paths.append(Path(downloaded))
    return paths


def discover_local_parquets(parquet_dir: Path) -> list[Path]:
    return sorted(parquet_dir.glob("*.parquet"), key=lambda path: parquet_sort_key(path.name))


def is_spanish(value: object) -> bool:
    if value is None:
        return True
    normalized = str(value).strip().lower().replace("_", "-")
    if not normalized:
        return True
    return (
        normalized == "es"
        or normalized.startswith("es-")
        or normalized.startswith("spanish")
        or normalized.startswith("spa")
    )


def iter_books(path: Path, batch_size: int = 16) -> Iterator[tuple[str, str]]:
    parquet = require_pyarrow()
    parquet_file = parquet.ParquetFile(path)
    columns = parquet_file.schema_arrow.names
    text_column = next((name for name in TEXT_COLUMNS if name in columns), None)
    if text_column is None:
        raise ValueError(f"No text column found in {path}; columns={columns}")
    language_column = next(
        (name for name in LANGUAGE_COLUMNS if name in columns),
        None,
    )
    identifier_column = next(
        (name for name in ("identifier", "id", "title") if name in columns),
        None,
    )
    selected_columns = [text_column]
    for optional in (language_column, identifier_column):
        if optional and optional not in selected_columns:
            selected_columns.append(optional)

    row_number = 0
    for batch in parquet_file.iter_batches(
        batch_size=batch_size,
        columns=selected_columns,
    ):
        rows = batch.to_pylist()
        for row in rows:
            row_number += 1
            text = row.get(text_column)
            if not isinstance(text, str) or not text.strip():
                continue
            if language_column and not is_spanish(row.get(language_column)):
                continue
            identifier = str(row.get(identifier_column) or row_number)
            yield identifier, text


def count_book_words(text: str, min_word_length: int) -> tuple[int, Counter[str]]:
    counts: Counter[str] = Counter()
    token_count = 0
    for match in WORD_RE.finditer(text):
        token_count += 1
        word = normalize_target_word(match.group(0))
        if word and len(word) >= min_word_length:
            counts[word] += 1
    return token_count, counts


def write_frequency(path: Path, counts: Counter[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for rank, (word, count) in enumerate(
            sorted(counts.items(), key=lambda item: (-item[1], item[0])),
            start=1,
        ):
            handle.write(f"{rank}\t{word}\t{count}\n")


def safe_identifier(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_")
    return cleaned[:80] or "book"


def prepare_frequency_split(
    paths: list[Path],
    *,
    output: Path,
    min_word_length: int,
    min_book_words: int,
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    books = 0
    tokens = 0
    for shard_index, path in enumerate(paths, start=1):
        print(f"[frequency {shard_index}/{len(paths)}] {path.name}", flush=True)
        for _identifier, text in iter_books(path):
            book_tokens, book_counts = count_book_words(text, min_word_length)
            if book_tokens < min_book_words:
                continue
            books += 1
            tokens += book_tokens
            counts.update(book_counts)
    write_frequency(output, counts)
    return {"books": books, "tokens": tokens, "types": len(counts)}


def prepare_context_split(
    paths: list[Path],
    *,
    output_dir: Path,
    min_book_words: int,
) -> dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    books = 0
    tokens = 0
    for shard_index, path in enumerate(paths, start=1):
        shard_dir = output_dir / path.stem
        shard_dir.mkdir(parents=True, exist_ok=True)
        print(f"[contexts {shard_index}/{len(paths)}] {path.name}", flush=True)
        for row_index, (identifier, text) in enumerate(iter_books(path), start=1):
            book_tokens = len(WORD_RE.findall(text))
            if book_tokens < min_book_words:
                continue
            books += 1
            tokens += book_tokens
            filename = f"{row_index:05d}-{safe_identifier(identifier)}.txt"
            destination = shard_dir / filename
            if not destination.exists():
                destination.write_text(text, encoding="utf-8")
    return {"books": books, "tokens": tokens}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument("--revision", default="main")
    parser.add_argument(
        "--parquet-dir",
        type=Path,
        default=ROOT / "data/raw/spanish_pd_books/parquet",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data/processed/spanish_pd_books",
    )
    parser.add_argument(
        "--frequency-shards",
        type=int,
        default=0,
        help=(
            "Optional number of disjoint Parquet shards used to derive word "
            "frequencies. Use 0 when supplying an external frequency list."
        ),
    )
    parser.add_argument(
        "--context-shards",
        type=int,
        default=4,
        help="Number of following Parquet shards exported as context books.",
    )
    parser.add_argument(
        "--start-shard",
        type=int,
        default=0,
        help="Zero-based offset into the numerically sorted Parquet list.",
    )
    parser.add_argument("--min-word-length", type=int, default=3)
    parser.add_argument("--min-book-words", type=int, default=500)
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Use Parquet files already present in --parquet-dir; do not download.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    total_shards = args.frequency_shards + args.context_shards
    if args.frequency_shards < 0 or args.context_shards < 1:
        raise SystemExit(
            "--frequency-shards must be non-negative and --context-shards positive"
        )
    if args.start_shard < 0:
        raise SystemExit("--start-shard must be non-negative")

    if args.local_only:
        available = discover_local_parquets(args.parquet_dir)
        selected_paths = available[args.start_shard : args.start_shard + total_shards]
    else:
        available_names = discover_remote_parquets(args.repo_id, args.revision)
        selected_names = available_names[
            args.start_shard : args.start_shard + total_shards
        ]
        selected_paths = download_parquets(
            selected_names,
            repo_id=args.repo_id,
            revision=args.revision,
            parquet_dir=args.parquet_dir,
        )
    if len(selected_paths) != total_shards:
        raise SystemExit(
            f"Found {len(selected_paths)} usable shards; expected {total_shards}"
        )

    frequency_paths = selected_paths[: args.frequency_shards]
    context_paths = selected_paths[args.frequency_shards :]
    frequency_output = args.output_dir / "frequency.tsv"
    context_output = args.output_dir / "contexts"
    frequency_stats = None
    if frequency_paths:
        frequency_stats = prepare_frequency_split(
            frequency_paths,
            output=frequency_output,
            min_word_length=args.min_word_length,
            min_book_words=args.min_book_words,
        )
    context_stats = prepare_context_split(
        context_paths,
        output_dir=context_output,
        min_book_words=args.min_book_words,
    )
    manifest = {
        "repo_id": args.repo_id,
        "revision": args.revision,
        "frequency_shards": [path.name for path in frequency_paths],
        "context_shards": [path.name for path in context_paths],
        "frequency": frequency_stats,
        "contexts": context_stats,
        "min_word_length": args.min_word_length,
        "min_book_words": args.min_book_words,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if frequency_paths:
        print(f"Wrote frequency list to {frequency_output}")
    print(f"Wrote context books to {context_output}")


if __name__ == "__main__":
    main()
