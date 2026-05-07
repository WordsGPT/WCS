"""Dataset preparation for the Word Coverage Score paper.

This module implements the paper's frequency-based lexical selection and
contextual pairing steps without depending on external dataset libraries.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence


WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]*")


@dataclass(frozen=True)
class FrequencyEntry:
    rank: int
    word: str
    count: int | None = None


@dataclass(frozen=True)
class CorpusMatch:
    prefix: str
    matched_text: str
    source_path: str
    match_start_char: int
    match_end_char: int
    context_token_count: int
    search_start_char: int


@dataclass(frozen=True)
class IndexedOccurrence:
    word: str
    prefix: str
    matched_text: str
    source_path: str
    match_start_char: int
    match_end_char: int
    context_token_count: int
    global_start_char: int


@dataclass(frozen=True)
class Sample:
    id: str
    word: str
    rank: int
    count: int | None
    prefix: str
    matched_text: str
    source_path: str
    match_start_char: int
    match_end_char: int
    context_token_count: int
    search_start_char: int
    metadata: dict[str, int | str]


def load_frequency_entries(path: Path) -> list[FrequencyEntry]:
    entries: list[FrequencyEntry] = []
    with path.open("r", encoding="utf-8") as handle:
        for row_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parsed = parse_frequency_line(line, fallback_rank=len(entries) + 1)
            if parsed is None:
                raise ValueError(f"Could not parse frequency row {row_number}: {line!r}")
            entries.append(parsed)
    return entries


def parse_frequency_line(line: str, fallback_rank: int) -> FrequencyEntry | None:
    if "," in line:
        fields = next(csv.reader([line]))
    elif "\t" in line:
        fields = line.split("\t")
    else:
        fields = line.split()

    fields = [field.strip() for field in fields if field.strip()]
    if not fields:
        return None

    lowered = [field.lower() for field in fields]
    if "word" in lowered and ("rank" in lowered or "count" in lowered):
        return None

    rank: int | None = None
    count: int | None = None
    word: str | None = None

    if len(fields) == 1:
        word = fields[0]
    elif len(fields) == 2:
        if fields[0].isdigit():
            rank = int(fields[0])
            word = fields[1]
        elif fields[1].isdigit():
            word = fields[0]
            count = int(fields[1])
        else:
            word = fields[0]
    else:
        if fields[0].isdigit():
            rank = int(fields[0])
            word = fields[1]
            count = int(fields[2]) if fields[2].isdigit() else None
        else:
            word = fields[0]
            count = int(fields[1]) if fields[1].isdigit() else None

    if word is None:
        return None
    cleaned = normalize_target_word(word)
    if not cleaned:
        return None
    return FrequencyEntry(rank=rank or fallback_rank, word=cleaned, count=count)


def normalize_target_word(word: str) -> str:
    word = word.strip().lower()
    match = WORD_RE.fullmatch(word)
    return match.group(0) if match else ""


def select_rank_band(
    entries: Sequence[FrequencyEntry],
    rank_min: int,
    rank_max: int,
    sample_size: int,
    seed: int,
    min_word_length: int = 1,
    allowed_words: set[str] | None = None,
) -> list[FrequencyEntry]:
    if rank_min > rank_max:
        raise ValueError("--rank-min must be less than or equal to --rank-max")
    band = [
        entry
        for entry in entries
        if rank_min <= entry.rank <= rank_max and len(entry.word) >= min_word_length
        and (allowed_words is None or entry.word in allowed_words)
    ]
    if sample_size <= 0 or sample_size >= len(band):
        selected = list(band)
    else:
        rng = random.Random(seed)
        selected = rng.sample(band, sample_size)
    return sorted(selected, key=lambda entry: entry.rank)


def load_dictionary(path: Path) -> set[str]:
    words: set[str] = set()
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            word = normalize_target_word(line.strip())
            if word:
                words.add(word)
    return words


def iter_corpus_files(corpus_path: Path) -> Iterator[Path]:
    if corpus_path.is_file():
        yield corpus_path
        return
    for suffix in ("*.txt", "*.text"):
        yield from sorted(corpus_path.rglob(suffix))


def find_context_for_word(
    word: str,
    corpus_files: Sequence[Path],
    context_tokens: int,
    rng: random.Random,
) -> CorpusMatch | None:
    files = list(corpus_files)
    rng.shuffle(files)
    pattern = re.compile(rf"\b{re.escape(word)}\b", flags=re.IGNORECASE)

    for path in files:
        text = path.read_text(encoding="utf-8", errors="ignore")
        if len(text) == 0:
            continue
        search_start = rng.randrange(len(text))
        match = pattern.search(text, pos=search_start) or pattern.search(text, pos=0)
        while match is not None:
            prefix = prefix_before(text, match.start(), context_tokens)
            token_count = count_tokens(prefix)
            if token_count >= context_tokens:
                return CorpusMatch(
                    prefix=prefix,
                    matched_text=text[match.start() : match.end()],
                    source_path=str(path),
                    match_start_char=match.start(),
                    match_end_char=match.end(),
                    context_token_count=token_count,
                    search_start_char=search_start,
                )
            match = pattern.search(text, pos=match.end())
    return None


def index_corpus_occurrences(
    words: set[str],
    corpus_files: Sequence[Path],
    context_tokens: int,
    exclude_capitalized_matches: bool = False,
) -> tuple[dict[str, list[IndexedOccurrence]], int]:
    occurrences: dict[str, list[IndexedOccurrence]] = {word: [] for word in words}
    global_offset = 0

    for path in sorted(corpus_files):
        text = path.read_text(encoding="utf-8", errors="ignore")
        token_matches = list(WORD_RE.finditer(text))
        for token_index, match in enumerate(token_matches):
            normalized = normalize_target_word(match.group(0))
            if normalized not in words or token_index < context_tokens:
                continue
            if exclude_capitalized_matches and match.group(0)[:1].isupper():
                continue
            prefix_tokens = token_matches[token_index - context_tokens : token_index]
            prefix = " ".join(token.group(0) for token in prefix_tokens)
            occurrences[normalized].append(
                IndexedOccurrence(
                    word=normalized,
                    prefix=prefix,
                    matched_text=match.group(0),
                    source_path=str(path),
                    match_start_char=match.start(),
                    match_end_char=match.end(),
                    context_token_count=context_tokens,
                    global_start_char=global_offset + match.start(),
                )
            )
        global_offset += len(text) + 1

    return occurrences, global_offset


def choose_occurrence_after_offset(
    occurrences: Sequence[IndexedOccurrence],
    search_start_char: int,
) -> IndexedOccurrence | None:
    if not occurrences:
        return None
    for occurrence in sorted(occurrences, key=lambda item: item.global_start_char):
        if occurrence.global_start_char >= search_start_char:
            return occurrence
    return min(occurrences, key=lambda item: item.global_start_char)


def prefix_before(text: str, char_offset: int, context_tokens: int) -> str:
    prefix_text = text[:char_offset]
    tokens = list(WORD_RE.finditer(prefix_text))
    if context_tokens > 0:
        tokens = tokens[-context_tokens:]
    if not tokens:
        return ""
    return " ".join(token.group(0) for token in tokens)


def count_tokens(text: str) -> int:
    return len(WORD_RE.findall(text))


def build_samples(
    frequency_path: Path,
    corpus_path: Path,
    rank_min: int,
    rank_max: int,
    sample_size: int,
    context_tokens: int,
    seed: int,
    exclude_capitalized_matches: bool = False,
    min_word_length: int = 1,
    dictionary_path: Path | None = None,
) -> tuple[list[Sample], list[FrequencyEntry]]:
    entries = load_frequency_entries(frequency_path)
    allowed_words = load_dictionary(dictionary_path) if dictionary_path else None
    selected = select_rank_band(
        entries,
        rank_min,
        rank_max,
        sample_size=0,
        seed=seed,
        min_word_length=min_word_length,
        allowed_words=allowed_words,
    )
    rng = random.Random(seed)
    rng.shuffle(selected)
    corpus_files = list(iter_corpus_files(corpus_path))
    if not corpus_files:
        raise FileNotFoundError(f"No .txt or .text files found under {corpus_path}")

    samples: list[Sample] = []
    missing: list[FrequencyEntry] = []
    occurrence_index, corpus_char_count = index_corpus_occurrences(
        words={entry.word for entry in selected},
        corpus_files=corpus_files,
        context_tokens=context_tokens,
        exclude_capitalized_matches=exclude_capitalized_matches,
    )

    for entry in selected:
        if sample_size > 0 and len(samples) >= sample_size:
            break
        search_start = rng.randrange(corpus_char_count) if corpus_char_count > 0 else 0
        occurrence = choose_occurrence_after_offset(occurrence_index[entry.word], search_start)
        if occurrence is None:
            missing.append(entry)
            continue
        samples.append(
            Sample(
                id=f"sample-{len(samples) + 1:06d}",
                word=entry.word,
                rank=entry.rank,
                count=entry.count,
                prefix=occurrence.prefix,
                matched_text=occurrence.matched_text,
                source_path=occurrence.source_path,
                match_start_char=occurrence.match_start_char,
                match_end_char=occurrence.match_end_char,
                context_token_count=occurrence.context_token_count,
                search_start_char=search_start,
                metadata={
                    "rank_min": rank_min,
                    "rank_max": rank_max,
                    "sample_size": sample_size,
                    "context_tokens": context_tokens,
                    "seed": seed,
                    "selection": "filled_from_rank_band",
                    "exclude_capitalized_matches": int(exclude_capitalized_matches),
                    "min_word_length": min_word_length,
                    "dictionary": str(dictionary_path) if dictionary_path else "",
                },
            )
        )
    return samples, missing


def write_jsonl(samples: Iterable[Sample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(asdict(sample), ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build WCS target/context samples.")
    parser.add_argument("--frequency", type=Path, required=True, help="Ranked frequency list.")
    parser.add_argument("--corpus", type=Path, required=True, help="Corpus file or directory of text files.")
    parser.add_argument("--output", type=Path, default=Path("data/processed/samples.jsonl"))
    parser.add_argument("--rank-min", type=int, default=10_000)
    parser.add_argument("--rank-max", type=int, default=40_000)
    parser.add_argument("--sample-size", type=int, default=1_000)
    parser.add_argument("--context-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--min-word-length",
        type=int,
        default=3,
        help="Minimum normalized word length to keep from the frequency list.",
    )
    parser.add_argument(
        "--dictionary",
        type=Path,
        default=None,
        help="Optional newline-delimited dictionary. Targets must appear in it.",
    )
    parser.add_argument(
        "--exclude-capitalized-matches",
        action="store_true",
        help="Skip corpus matches whose surface form starts with a capital letter.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples, missing = build_samples(
        frequency_path=args.frequency,
        corpus_path=args.corpus,
        rank_min=args.rank_min,
        rank_max=args.rank_max,
        sample_size=args.sample_size,
        context_tokens=args.context_tokens,
        seed=args.seed,
        exclude_capitalized_matches=args.exclude_capitalized_matches,
        min_word_length=args.min_word_length,
        dictionary_path=args.dictionary,
    )
    write_jsonl(samples, args.output)
    print(f"Wrote {len(samples)} samples to {args.output}")
    if missing:
        print(f"Skipped {len(missing)} words with no full-context match")


if __name__ == "__main__":
    main()
