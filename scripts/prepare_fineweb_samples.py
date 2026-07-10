#!/usr/bin/env python
"""Stream FineWeb and build Gemini-checked WCS samples without a 28 GB download."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from wcs.dataset_builder import (
    ContextDecision,
    FrequencyEntry,
    IndexedOccurrence,
    Sample,
    WORD_RE,
    load_frequency_entries,
    validate_contexts_with_gemini_detailed,
)


def select_entries(args: argparse.Namespace) -> list[FrequencyEntry]:
    entries = [
        entry
        for entry in load_frequency_entries(args.frequency)
        if args.rank_min <= entry.rank <= args.rank_max
        and len(entry.word) >= args.min_word_length
    ]
    random.Random(args.seed).shuffle(entries)
    wanted = args.sample_size * args.candidate_pool_multiplier
    if len(entries) < wanted:
        raise SystemExit(f"Frequency band has only {len(entries)} eligible words; need {wanted}.")
    return entries[:wanted]


def occurrence_from_match(
    text: str, matches: list[Any], index: int, row: dict[str, Any]
) -> IndexedOccurrence | None:
    if index < 1:
        return None
    match = matches[index]
    context_start = max(0, index - occurrence_from_match.context_words)
    if index - context_start < occurrence_from_match.context_words:
        return None
    prefix = text[matches[context_start].start() : matches[index - 1].end()]
    raw_excerpt = text[matches[context_start].start() : match.end()]
    source_id = str(row.get("id") or row.get("url") or "unknown")
    return IndexedOccurrence(
        word=match.group(0).casefold(),
        prefix=prefix,
        raw_excerpt=raw_excerpt,
        matched_text=match.group(0),
        source_path=f"fineweb://{source_id}",
        match_start_char=match.start(),
        match_end_char=match.end(),
        context_token_count=occurrence_from_match.context_words,
        global_start_char=match.start(),
    )


occurrence_from_match.context_words = 256


def write_candidate_checkpoint(
    path: Path, candidates: dict[str, list[IndexedOccurrence]], entries: list[FrequencyEntry]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rank_by_word = {entry.word: entry.rank for entry in entries}
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for word, occurrences in candidates.items():
            for occurrence in occurrences:
                row = asdict(occurrence)
                row["rank"] = rank_by_word[word]
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def load_candidate_checkpoint(path: Path) -> dict[str, list[IndexedOccurrence]]:
    candidates: dict[str, list[IndexedOccurrence]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            row.pop("rank", None)
            occurrence = IndexedOccurrence(**row)
            candidates.setdefault(occurrence.word, []).append(occurrence)
    return candidates


def collect_candidates(
    args: argparse.Namespace, entries: list[FrequencyEntry]
) -> dict[str, list[IndexedOccurrence]]:
    if args.candidates.exists() and not args.refresh_candidates:
        print(f"[resume] loading candidate cache {args.candidates}", flush=True)
        return load_candidate_checkpoint(args.candidates)

    from datasets import load_dataset

    target_words = {entry.word for entry in entries}
    candidates: dict[str, list[IndexedOccurrence]] = {word: [] for word in target_words}
    partial_path = args.candidates.with_suffix(args.candidates.suffix + ".partial")
    if partial_path.exists() and not args.refresh_candidates:
        cached = load_candidate_checkpoint(partial_path)
        for word in target_words:
            candidates[word] = cached.get(word, [])[: args.candidate_contexts_per_word]
        print(f"[resume] continuing candidate scan from {partial_path}", flush=True)
    seen = {
        word: {(item.source_path, item.match_start_char) for item in occurrences}
        for word, occurrences in candidates.items()
    }
    occurrence_from_match.context_words = args.context_words
    dataset = load_dataset(
        args.dataset,
        name=args.config,
        split="train",
        streaming=True,
        columns=["text", "id", "url", "dump", "language_score", "token_count"],
    )
    if args.shuffle_buffer > 0:
        dataset = dataset.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)

    full_words = sum(
        len(bucket) >= args.candidate_contexts_per_word for bucket in candidates.values()
    )
    for document_index, row in enumerate(dataset, start=1):
        if document_index > args.max_documents:
            break
        text = str(row.get("text") or "")
        matches = list(WORD_RE.finditer(text))
        for index, match in enumerate(matches):
            word = match.group(0).casefold()
            bucket = candidates.get(word)
            if bucket is None or len(bucket) >= args.candidate_contexts_per_word:
                continue
            if args.exclude_capitalized_matches and match.group(0)[:1].isupper():
                continue
            occurrence = occurrence_from_match(text, matches, index, row)
            if occurrence is None:
                continue
            occurrence_key = (occurrence.source_path, occurrence.match_start_char)
            if occurrence_key in seen[word]:
                continue
            bucket.append(occurrence)
            seen[word].add(occurrence_key)
            if len(bucket) == args.candidate_contexts_per_word:
                full_words += 1
        if document_index % args.progress_documents == 0:
            usable = sum(len(values) >= args.contexts_per_word for values in candidates.values())
            print(
                f"[fineweb] documents={document_index:,} words_with_{args.contexts_per_word}+={usable} "
                f"full_candidate_buckets={full_words}",
                flush=True,
            )
            write_candidate_checkpoint(partial_path, candidates, entries)
        if full_words == len(candidates):
            print(f"[fineweb] all {full_words} candidate buckets filled", flush=True)
            break
    write_candidate_checkpoint(args.candidates, candidates, entries)
    if partial_path.exists():
        partial_path.unlink()
    return candidates


def load_existing_samples(path: Path, contexts_per_word: int) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["word"]] = counts.get(row["word"], 0) + 1
    complete = {word for word, count in counts.items() if count == contexts_per_word}
    return [row for row in rows if row["word"] in complete]


def validate_and_write(
    args: argparse.Namespace,
    entries: list[FrequencyEntry],
    candidates: dict[str, list[IndexedOccurrence]],
) -> None:
    existing = load_existing_samples(args.output, args.contexts_per_word) if args.resume else []
    completed_words = {row["word"] for row in existing}
    if existing:
        print(f"[resume] preserving {len(completed_words)} complete word groups", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Also removes any word group interrupted partway through its atomic append.
    with args.output.open("w", encoding="utf-8") as handle:
        for row in existing:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    jobs = [
        entry for entry in entries
        if entry.word not in completed_words
        and len(candidates.get(entry.word, [])) >= args.contexts_per_word
    ]

    def check(entry: FrequencyEntry) -> tuple[FrequencyEntry, list[IndexedOccurrence], list[ContextDecision]]:
        occurrences = candidates[entry.word][: args.candidate_contexts_per_word]
        if args.skip_coherence_check:
            decisions = [ContextDecision(True, "accepted", "Coherence check skipped.")] * len(occurrences)
        else:
            decisions = validate_contexts_with_gemini_detailed(
                [occurrence.raw_excerpt for occurrence in occurrences],
                target_word=entry.word,
                model=args.coherence_model,
                language="English",
            )
        accepted = [occ for occ, decision in zip(occurrences, decisions) if decision.accepted]
        return entry, accepted[: args.contexts_per_word], decisions

    accepted_word_count = len(completed_words)
    with ThreadPoolExecutor(max_workers=args.coherence_workers) as executor:
        for start in range(0, len(jobs), args.coherence_workers):
            if accepted_word_count >= args.sample_size:
                break
            batch = jobs[start : start + args.coherence_workers]
            for entry, accepted, decisions in executor.map(check, batch):
                if accepted_word_count >= args.sample_size:
                    break
                if args.coherence_log:
                    args.coherence_log.parent.mkdir(parents=True, exist_ok=True)
                    with args.coherence_log.open("a", encoding="utf-8") as handle:
                        for occurrence, decision in zip(candidates[entry.word], decisions):
                            handle.write(json.dumps({
                                "word": entry.word,
                                "rank": entry.rank,
                                "accepted": decision.accepted,
                                "reason": decision.reason,
                                "note": decision.note,
                                "excerpt": occurrence.raw_excerpt,
                                "source_path": occurrence.source_path,
                            }, ensure_ascii=False) + "\n")
                if len(accepted) < args.contexts_per_word:
                    print(f"[reject] {entry.word}: accepted {len(accepted)}/{len(decisions)}", flush=True)
                    continue
                rows = []
                for occurrence in accepted:
                    sample_number = len(existing) + len(rows) + 1
                    sample = Sample(
                    id=f"sample-{sample_number:06d}",
                    word=entry.word,
                    rank=entry.rank,
                    count=entry.count,
                    prefix=occurrence.prefix,
                    matched_text=occurrence.matched_text,
                    source_path=occurrence.source_path,
                    match_start_char=occurrence.match_start_char,
                    match_end_char=occurrence.match_end_char,
                    context_token_count=occurrence.context_token_count,
                    search_start_char=0,
                    metadata={
                        "dataset": args.dataset,
                        "config": args.config,
                        "rank_min": args.rank_min,
                        "rank_max": args.rank_max,
                        "sample_size": args.sample_size,
                        "contexts_per_word": args.contexts_per_word,
                        "context_tokens": args.context_words,
                        "seed": args.seed,
                        "coherence_model": args.coherence_model,
                    },
                    )
                    rows.append(asdict(sample))
                with args.output.open("a", encoding="utf-8") as handle:
                    for row in rows:
                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                existing.extend(rows)
                accepted_word_count += 1
                print(f"[accept] {accepted_word_count}/{args.sample_size} {entry.word}", flush=True)
    if accepted_word_count < args.sample_size:
        raise SystemExit(
            f"Only built {accepted_word_count}/{args.sample_size} word groups. "
            "Increase --max-documents, --candidate-contexts-per-word, or the pool multiplier."
        )
    print(f"[done] wrote {len(existing)} samples to {args.output}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frequency", type=Path, default=ROOT / "data/raw/norvig_count_1w.txt")
    parser.add_argument("--output", type=Path, default=ROOT / "data/processed/samples.fineweb.jsonl")
    parser.add_argument("--candidates", type=Path, default=ROOT / "data/processed/fineweb_candidates.jsonl")
    parser.add_argument("--coherence-log", type=Path, default=ROOT / "logs/fineweb_coherence.jsonl")
    parser.add_argument("--dataset", default="HuggingFaceFW/fineweb")
    parser.add_argument("--config", default="sample-10BT")
    parser.add_argument("--rank-min", type=int, default=10_000)
    parser.add_argument("--rank-max", type=int, default=40_000)
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--contexts-per-word", type=int, default=50)
    parser.add_argument("--context-words", type=int, default=256)
    parser.add_argument("--candidate-contexts-per-word", type=int, default=80)
    parser.add_argument("--candidate-pool-multiplier", type=int, default=4)
    parser.add_argument("--max-documents", type=int, default=1_000_000)
    parser.add_argument("--shuffle-buffer", type=int, default=10_000)
    parser.add_argument("--progress-documents", type=int, default=25_000)
    parser.add_argument("--coherence-workers", type=int, default=12)
    parser.add_argument("--coherence-model", default="gemini-2.5-flash-lite")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--min-word-length", type=int, default=3)
    parser.add_argument("--exclude-capitalized-matches", action="store_true", default=True)
    parser.add_argument("--skip-coherence-check", action="store_true")
    parser.add_argument("--refresh-candidates", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.candidate_contexts_per_word < args.contexts_per_word:
        raise SystemExit("--candidate-contexts-per-word must be >= --contexts-per-word")
    entries = select_entries(args)
    candidates = collect_candidates(args, entries)
    validate_and_write(args, entries, candidates)


if __name__ == "__main__":
    main()
