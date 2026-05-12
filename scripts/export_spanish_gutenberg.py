#!/usr/bin/env python
"""Export long Spanish books from Project Gutenberg metadata.

This is intentionally diagnostic-heavy because several Gutenberg datasets expose
small preview/config slices that look valid but contain no Spanish books.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


DEFAULT_DATASET = "zkeown/gutenberg-corpus"
DEFAULT_CONFIG_CANDIDATES = ("books", "default")
DEFAULT_GUTENDEX_URL = "https://gutendex.com/books/"
SPANISH_LANGUAGE_VALUES = {"es", "spa", "spanish", "espanol", "español"}
GUTENBERG_START_RE = re.compile(r"\*\*\* START OF (?:THE )?PROJECT GUTENBERG EBOOK.*?\*\*\*", re.I | re.S)
GUTENBERG_END_RE = re.compile(r"\*\*\* END OF (?:THE )?PROJECT GUTENBERG EBOOK.*", re.I | re.S)


def normalize_language(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        return " ".join(normalize_language(item) for item in value)
    return str(value).strip().lower()


def is_spanish(value: Any) -> bool:
    normalized = normalize_language(value)
    if normalized in SPANISH_LANGUAGE_VALUES:
        return True
    parts = {part.strip() for part in re.split(r"[,;/| ]+", normalized) if part.strip()}
    return bool(parts & SPANISH_LANGUAGE_VALUES)


def safe_filename(value: Any, fallback: str) -> str:
    raw = str(value or fallback)
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", raw).strip("_")
    return cleaned[:80] or fallback


def fetch_json(url: str, timeout: int, retries: int) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except (TimeoutError, urllib.error.URLError, json.JSONDecodeError, OSError) as error:
            last_error = error
            if attempt < retries:
                time.sleep(min(2 ** attempt, 20))
    raise RuntimeError(f"Failed to fetch JSON from {url}: {last_error}")


def fetch_text(url: str, timeout: int, retries: int) -> str:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                raw = response.read()
                content_type = response.headers.get_content_charset()
                encoding = content_type or "utf-8"
                return raw.decode(encoding, errors="replace")
        except (TimeoutError, urllib.error.URLError, UnicodeError) as error:
            last_error = error
            if attempt < retries:
                time.sleep(min(2 ** attempt, 20))
    raise RuntimeError(f"Failed to fetch text from {url}: {last_error}")


def strip_gutenberg_boilerplate(text: str) -> str:
    start = GUTENBERG_START_RE.search(text)
    if start:
        text = text[start.end() :]
    end = GUTENBERG_END_RE.search(text)
    if end:
        text = text[: end.start()]
    return text.strip()


def gutendex_first_url(base_url: str, languages: str, mime_type: str, copyright: str) -> str:
    query = urllib.parse.urlencode(
        {
            "languages": languages,
            "mime_type": mime_type,
            "copyright": copyright,
            "sort": "ascending",
        }
    )
    return f"{base_url.rstrip('/')}/?{query}"


def select_plain_text_url(formats: dict[str, Any]) -> str | None:
    preferred_keys = [
        "text/plain; charset=utf-8",
        "text/plain; charset=us-ascii",
        "text/plain",
    ]
    for key in preferred_keys:
        value = formats.get(key)
        if isinstance(value, str) and value:
            return value
    for key, value in formats.items():
        if key.startswith("text/plain") and isinstance(value, str) and value:
            return value
    return None


def author_label(book: dict[str, Any]) -> str:
    authors = book.get("authors") or []
    if not authors:
        return "unknown_author"
    first = authors[0]
    if isinstance(first, dict):
        return str(first.get("name") or "unknown_author")
    return str(first)


def inspect_gutendex(args: argparse.Namespace) -> tuple[int, int, Counter[str], Counter[str]]:
    url = gutendex_first_url(args.gutendex_url, args.languages, args.mime_type, args.copyright)
    pages = 0
    books = 0
    languages: Counter[str] = Counter()
    format_keys: Counter[str] = Counter()

    while url and pages < args.inspect_pages:
        data = fetch_json(url, args.timeout, args.retries)
        pages += 1
        for book in data.get("results", []):
            books += 1
            languages.update(normalize_language(lang) or "<missing>" for lang in book.get("languages", []))
            format_keys.update((book.get("formats") or {}).keys())
        url = data.get("next")

    return pages, books, languages, format_keys


def export_from_gutendex(args: argparse.Namespace) -> int:
    first_url = gutendex_first_url(args.gutendex_url, args.languages, args.mime_type, args.copyright)
    print(f"Gutendex query: {first_url}", flush=True)
    pages, books, languages, format_keys = inspect_gutendex(args)
    print(f"Inspected {books} books across {pages} page(s)", flush=True)
    print_counter("Language values in inspected Gutendex results:", languages)
    print_counter("Format MIME types in inspected Gutendex results:", format_keys)

    if args.inspect_only:
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    url = first_url
    page = 0
    scanned = 0
    written = 0
    skip_reasons: Counter[str] = Counter()
    downloaded_chars = 0

    while url:
        if args.max_pages is not None and page >= args.max_pages:
            break
        data = fetch_json(url, args.timeout, args.retries)
        page += 1
        for book in data.get("results", []):
            scanned += 1
            if not is_spanish(book.get("languages", [])):
                skip_reasons["non_spanish_language"] += 1
                continue
            text_url = select_plain_text_url(book.get("formats") or {})
            if not text_url:
                skip_reasons["missing_plain_text_format"] += 1
                continue

            try:
                text = strip_gutenberg_boilerplate(fetch_text(text_url, args.timeout, args.retries))
            except RuntimeError as error:
                skip_reasons["download_error"] += 1
                print(f"Download failed for book {book.get('id')}: {error}", flush=True)
                continue

            downloaded_chars += len(text)
            if len(text) < args.min_chars:
                skip_reasons["short_text_after_cleaning"] += 1
                continue

            book_id = safe_filename(book.get("id"), f"book_{written + 1:06d}")
            title = safe_filename(book.get("title"), "untitled")
            author = safe_filename(author_label(book), "unknown_author")
            path = args.output_dir / f"{book_id}_{author}_{title}.txt"
            path.write_text(text, encoding="utf-8")
            written += 1
            if args.progress_interval > 0 and written % args.progress_interval == 0:
                print(
                    f"Wrote {written} books after scanning {scanned} Gutendex records "
                    f"({downloaded_chars:,} downloaded chars)",
                    flush=True,
                )
            if args.max_books is not None and written >= args.max_books:
                break
        if args.max_books is not None and written >= args.max_books:
            break
        url = data.get("next")

    print(f"\nScanned {scanned} Gutendex records", flush=True)
    print(f"Wrote {written} Spanish books to {args.output_dir}", flush=True)
    print_counter("Skip reasons:", skip_reasons)
    if written == 0:
        print("No books were written.", flush=True)
        print("What to check next:", flush=True)
        print("  1. Run with --inspect-only and inspect formats/languages.", flush=True)
        print("  2. Lower --min-chars if books are present but too short after cleaning.", flush=True)
        print("  3. Increase --max-pages or remove it if you limited pagination.", flush=True)
        return 2
    return 0


def import_datasets() -> tuple[Any, Any]:
    try:
        from datasets import get_dataset_config_names, load_dataset
    except ImportError as error:
        raise SystemExit(
            "Missing dependency: datasets. Install it on the server with:\n"
            "  .venv/bin/pip install datasets"
        ) from error
    return get_dataset_config_names, load_dataset


def config_options(raw: str | None) -> list[str | None]:
    if raw:
        return [None if item.strip().lower() == "none" else item.strip() for item in raw.split(",") if item.strip()]
    return list(DEFAULT_CONFIG_CANDIDATES) + [None]


def load_stream(load_dataset: Any, dataset: str, config: str | None, split: str) -> Iterable[dict[str, Any]]:
    if config is None:
        return load_dataset(dataset, split=split, streaming=True)
    return load_dataset(dataset, config, split=split, streaming=True)


def inspect_stream(rows: Iterable[dict[str, Any]], limit: int) -> tuple[Counter[str], list[str], int]:
    languages: Counter[str] = Counter()
    fields: set[str] = set()
    seen = 0
    for row in rows:
        seen += 1
        fields.update(row.keys())
        languages[normalize_language(row.get("language")) or "<missing>"] += 1
        if seen >= limit:
            break
    return languages, sorted(fields), seen


def write_books(
    rows: Iterable[dict[str, Any]],
    output_dir: Path,
    min_chars: int,
    max_books: int | None,
    progress_interval: int,
) -> tuple[int, int, Counter[str], Counter[str]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    scanned = 0
    languages: Counter[str] = Counter()
    skip_reasons: Counter[str] = Counter()

    for row in rows:
        scanned += 1
        language = row.get("language")
        languages[normalize_language(language) or "<missing>"] += 1
        if not is_spanish(language):
            skip_reasons["non_spanish_language"] += 1
            continue

        text = row.get("text") or ""
        if not isinstance(text, str) or len(text) < min_chars:
            skip_reasons["short_or_missing_text"] += 1
            continue

        book_id = safe_filename(row.get("id"), f"book_{written + 1:06d}")
        title = safe_filename(row.get("title"), "untitled")
        path = output_dir / f"{book_id}_{title}.txt"
        path.write_text(text, encoding="utf-8")
        written += 1

        if progress_interval > 0 and written % progress_interval == 0:
            print(f"Wrote {written} books after scanning {scanned} rows", flush=True)
        if max_books is not None and written >= max_books:
            break

    return written, scanned, languages, skip_reasons


def print_counter(title: str, counter: Counter[str], limit: int = 20) -> None:
    print(title)
    if not counter:
        print("  <none>")
        return
    for key, count in counter.most_common(limit):
        print(f"  {key}: {count}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export long Spanish books from Project Gutenberg metadata."
    )
    parser.add_argument("--source", choices=["gutendex", "hf"], default="gutendex")
    parser.add_argument("--gutendex-url", default=DEFAULT_GUTENDEX_URL)
    parser.add_argument("--languages", default="es")
    parser.add_argument("--mime-type", default="text/plain")
    parser.add_argument("--copyright", default="false")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument(
        "--configs",
        default=None,
        help=(
            "Comma-separated configs to try. Use 'none' to load without a config. "
            "Default tries books,default,none."
        ),
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw/spanish_gutenberg"))
    parser.add_argument("--min-chars", type=int, default=100_000)
    parser.add_argument("--max-books", type=int, default=None)
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument("--inspect-rows", type=int, default=10_000)
    parser.add_argument("--inspect-pages", type=int, default=2)
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--progress-interval", type=int, default=25)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.source == "gutendex":
        return export_from_gutendex(args)

    get_dataset_config_names, load_dataset = import_datasets()

    try:
        available_configs = get_dataset_config_names(args.dataset)
        print(f"Available configs for {args.dataset}: {available_configs}", flush=True)
    except Exception as error:  # noqa: BLE001 - diagnostics should report original failure.
        print(f"Warning: could not list configs for {args.dataset}: {error}", flush=True)
        available_configs = []

    last_error: Exception | None = None
    for config in config_options(args.configs):
        label = config if config is not None else "<no config>"
        print(f"\nTrying config {label}", flush=True)
        try:
            inspection_rows = load_stream(load_dataset, args.dataset, config, args.split)
            languages, fields, seen = inspect_stream(inspection_rows, args.inspect_rows)
            print(f"Inspected {seen} rows for config {label}", flush=True)
            print(f"Fields: {fields}", flush=True)
            print_counter("Language values in inspected rows:", languages)

            if args.inspect_only:
                continue

            if not any(is_spanish(language) for language in languages):
                print(
                    f"No Spanish language rows found in first {seen} rows for config {label}; "
                    "trying next config.",
                    flush=True,
                )
                continue

            export_rows = load_stream(load_dataset, args.dataset, config, args.split)
            written, scanned, all_languages, skip_reasons = write_books(
                export_rows,
                args.output_dir,
                args.min_chars,
                args.max_books,
                args.progress_interval,
            )
            print(f"\nScanned {scanned} rows for config {label}", flush=True)
            print(f"Wrote {written} Spanish books to {args.output_dir}", flush=True)
            print_counter("Languages seen during export:", all_languages)
            print_counter("Skip reasons:", skip_reasons)
            if written > 0:
                return 0
            print("No books were written for this config; trying next config.", flush=True)
        except Exception as error:  # noqa: BLE001 - continue through candidate configs.
            last_error = error
            print(f"Config {label} failed: {type(error).__name__}: {error}", flush=True)

    print("\nNo Spanish books were exported.", flush=True)
    print("What to check next:", flush=True)
    print("  1. Run with --inspect-only and inspect the language values.", flush=True)
    print("  2. Pass a specific --configs value from the available configs above.", flush=True)
    print("  3. Lower --min-chars if Spanish books are present but too short.", flush=True)
    if last_error is not None:
        print(f"Last error: {type(last_error).__name__}: {last_error}", flush=True)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
