#!/usr/bin/env python
"""Re-audit Spanish PD Books samples and replace only rejected contexts."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from functools import partial
from pathlib import Path
from typing import Callable, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from wcs.dataset_builder import (  # noqa: E402
    DEFAULT_COHERENCE_WORKERS,
    DEFAULT_GEMINI_MODEL,
    ContextDecision,
    IndexedOccurrence,
    Sample,
    index_corpus_occurrences,
    iter_corpus_files,
    validate_contexts_with_gemini_detailed,
)

Validator = Callable[..., list[ContextDecision]]
LocationKey = tuple[str, int, int]


def _canonical_source(path: str) -> str:
    normalized = path.replace("\\", "/")
    marker = "data/processed/"
    marker_index = normalized.find(marker)
    return normalized[marker_index:] if marker_index >= 0 else normalized


def _sample_key(sample: Sample) -> LocationKey:
    return (
        _canonical_source(sample.source_path),
        sample.match_start_char,
        sample.match_end_char,
    )


def _occurrence_key(occurrence: IndexedOccurrence) -> LocationKey:
    return (
        _canonical_source(occurrence.source_path),
        occurrence.match_start_char,
        occurrence.match_end_char,
    )


def _sample_excerpt(sample: Sample) -> str:
    return f"{sample.prefix.rstrip()} {sample.matched_text}".strip()


def load_samples(path: Path) -> list[Sample]:
    rows = [
        Sample(**json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"No samples found in {path}")
    ids = [row.id for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate sample IDs in {path}")
    return rows


def load_occurrences(path: Path) -> dict[str, list[IndexedOccurrence]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_occurrences = payload.get("occurrences")
    if not isinstance(raw_occurrences, dict):
        raise ValueError(f"Invalid occurrence index: {path}")
    return {
        str(word): [IndexedOccurrence(**item) for item in items]
        for word, items in raw_occurrences.items()
    }


def audit_existing_samples(
    samples: Sequence[Sample],
    *,
    validator: Validator = validate_contexts_with_gemini_detailed,
    model: str = DEFAULT_GEMINI_MODEL,
    workers: int = DEFAULT_COHERENCE_WORKERS,
) -> tuple[dict[str, ContextDecision], list[dict[str, object]]]:
    by_word: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        by_word[sample.word].append(sample)

    def audit_group(item: tuple[str, list[Sample]]) -> tuple[str, list[Sample], list[ContextDecision]]:
        word, group = item
        decisions = validator(
            [_sample_excerpt(sample) for sample in group],
            target_word=word,
            model=model,
            language="Spanish",
        )
        if len(decisions) != len(group):
            raise ValueError(
                f"Gemini returned {len(decisions)}/{len(group)} decisions for {word!r}"
            )
        return word, group, decisions

    decisions_by_id: dict[str, ContextDecision] = {}
    log_rows: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = executor.map(audit_group, by_word.items())
        for word, group, decisions in results:
            for sample, decision in zip(group, decisions):
                decisions_by_id[sample.id] = decision
                log_rows.append(
                    {
                        "stage": "existing",
                        "sample_id": sample.id,
                        "word": word,
                        "accepted": decision.accepted,
                        "reason": decision.reason,
                        "note": decision.note,
                        "selected_as_replacement": False,
                        "excerpt": _sample_excerpt(sample),
                        "source_path": sample.source_path,
                        "match_start_char": sample.match_start_char,
                        "match_end_char": sample.match_end_char,
                    }
                )
    return decisions_by_id, log_rows


def find_replacements(
    samples: Sequence[Sample],
    decisions_by_id: dict[str, ContextDecision],
    occurrences: dict[str, list[IndexedOccurrence]],
    *,
    validator: Validator = validate_contexts_with_gemini_detailed,
    model: str = DEFAULT_GEMINI_MODEL,
    workers: int = DEFAULT_COHERENCE_WORKERS,
    candidate_batch_size: int = 8,
) -> tuple[
    dict[str, tuple[IndexedOccurrence, ContextDecision]],
    list[dict[str, object]],
    dict[str, tuple[int, int]],
]:
    rejected_by_word: dict[str, list[Sample]] = defaultdict(list)
    used_keys = {_sample_key(sample) for sample in samples}
    for sample in samples:
        decision = decisions_by_id[sample.id]
        if not decision.accepted:
            rejected_by_word[sample.word].append(sample)

    candidate_jobs: list[tuple[str, list[IndexedOccurrence]]] = []
    for word in rejected_by_word:
        candidates: list[IndexedOccurrence] = []
        seen: set[LocationKey] = set()
        for occurrence in occurrences.get(word, []):
            key = _occurrence_key(occurrence)
            if key not in used_keys and key not in seen:
                candidates.append(occurrence)
                seen.add(key)
        candidate_jobs.append((word, candidates))

    def audit_candidates(
        job: tuple[str, list[IndexedOccurrence]],
    ) -> tuple[str, list[IndexedOccurrence], list[ContextDecision]]:
        word, candidates = job
        if not candidates:
            return word, candidates, []
        decisions: list[ContextDecision] = []
        for start in range(0, len(candidates), candidate_batch_size):
            batch = candidates[start : start + candidate_batch_size]
            decisions.extend(
                validator(
                    [candidate.raw_excerpt for candidate in batch],
                    target_word=word,
                    model=model,
                    language="Spanish",
                )
            )
        if len(decisions) != len(candidates):
            raise ValueError(
                f"Gemini returned {len(decisions)}/{len(candidates)} replacement "
                f"decisions for {word!r}"
            )
        return word, candidates, decisions

    replacements: dict[str, tuple[IndexedOccurrence, ContextDecision]] = {}
    log_rows: list[dict[str, object]] = []
    shortages: dict[str, tuple[int, int]] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = executor.map(audit_candidates, candidate_jobs)
        for word, candidates, decisions in results:
            accepted = [
                (candidate, decision)
                for candidate, decision in zip(candidates, decisions)
                if decision.accepted
            ]
            rejected_samples = rejected_by_word[word]
            selected_keys = {
                _occurrence_key(candidate)
                for candidate, _decision in accepted[: len(rejected_samples)]
            }
            for sample, replacement in zip(rejected_samples, accepted):
                replacements[sample.id] = replacement
            if len(accepted) < len(rejected_samples):
                shortages[word] = (len(rejected_samples), len(accepted))
            for candidate, decision in zip(candidates, decisions):
                log_rows.append(
                    {
                        "stage": "replacement_candidate",
                        "sample_id": None,
                        "word": word,
                        "accepted": decision.accepted,
                        "reason": decision.reason,
                        "note": decision.note,
                        "selected_as_replacement": (
                            _occurrence_key(candidate) in selected_keys
                        ),
                        "excerpt": candidate.raw_excerpt,
                        "source_path": candidate.source_path,
                        "match_start_char": candidate.match_start_char,
                        "match_end_char": candidate.match_end_char,
                    }
                )
    return replacements, log_rows, shortages


def apply_replacements(
    samples: Sequence[Sample],
    decisions_by_id: dict[str, ContextDecision],
    replacements: dict[str, tuple[IndexedOccurrence, ContextDecision]],
) -> list[Sample]:
    repaired: list[Sample] = []
    for sample in samples:
        if decisions_by_id[sample.id].accepted:
            repaired.append(sample)
            continue
        occurrence, replacement_decision = replacements[sample.id]
        original_decision = decisions_by_id[sample.id]
        metadata = dict(sample.metadata)
        metadata.update(
            {
                "quality_repaired": 1,
                "quality_repair_original_source_path": sample.source_path,
                "quality_repair_original_match_start_char": sample.match_start_char,
                "quality_repair_reason": original_decision.reason,
                "quality_repair_note": original_decision.note,
                "quality_replacement_note": replacement_decision.note,
            }
        )
        repaired.append(
            replace(
                sample,
                prefix=occurrence.prefix,
                matched_text=occurrence.matched_text,
                source_path=occurrence.source_path,
                match_start_char=occurrence.match_start_char,
                match_end_char=occurrence.match_end_char,
                context_token_count=occurrence.context_token_count,
                search_start_char=occurrence.global_start_char,
                metadata=metadata,
            )
        )
    return repaired


def _write_jsonl(path: Path, rows: Sequence[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            payload = asdict(row) if isinstance(row, Sample) else row
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Strictly re-audit the existing Spanish PD Books sample set and replace "
            "only rejected contexts from its cached occurrence index."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/samples.spanish_pd_books.100x10.jsonl"),
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=Path("data/processed/samples.spanish_pd_books.100x10.index.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/samples.spanish_pd_books.100x10.repaired.jsonl"),
    )
    parser.add_argument(
        "--audit-log",
        type=Path,
        default=Path("data/processed/spanish_pd_books.reaudit.jsonl"),
    )
    parser.add_argument("--model", default=DEFAULT_GEMINI_MODEL)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Concurrent Gemini requests. The conservative default avoids quota bursts.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=8,
        help="Attempts per Gemini request, with exponential backoff for HTTP 429/5xx.",
    )
    parser.add_argument(
        "--candidate-batch-size",
        type=int,
        default=8,
        help="Replacement alternatives per Gemini request; keep small to avoid truncation.",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("data/processed/spanish_pd_books/contexts"),
        help="Corpus used only when the cached index cannot fill a rejected context.",
    )
    parser.add_argument(
        "--fallback-candidates",
        type=int,
        default=200,
        help="Maximum targeted corpus occurrences retained per shortage word.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    if args.max_attempts < 1:
        raise SystemExit("--max-attempts must be at least 1")
    if args.candidate_batch_size < 1:
        raise SystemExit("--candidate-batch-size must be at least 1")
    if args.fallback_candidates < 1:
        raise SystemExit("--fallback-candidates must be at least 1")
    samples = load_samples(args.input)
    occurrences = load_occurrences(args.index)
    validator = partial(
        validate_contexts_with_gemini_detailed,
        max_attempts=args.max_attempts,
    )
    print(
        f"Re-auditing {len(samples)} contexts across "
        f"{len({sample.word for sample in samples})} Spanish target words...",
        flush=True,
    )
    decisions, audit_rows = audit_existing_samples(
        samples,
        validator=validator,
        model=args.model,
        workers=args.workers,
    )
    rejected = sum(not decision.accepted for decision in decisions.values())
    print(f"Gemini rejected {rejected}/{len(samples)} existing contexts.", flush=True)
    _write_jsonl(args.audit_log, audit_rows)
    replacements, candidate_rows, shortages = find_replacements(
        samples,
        decisions,
        occurrences,
        validator=validator,
        model=args.model,
        workers=args.workers,
        candidate_batch_size=args.candidate_batch_size,
    )
    if shortages and args.corpus.exists():
        shortage_words = set(shortages)
        corpus_files = list(iter_corpus_files(args.corpus))
        if corpus_files:
            print(
                "Cached alternatives were insufficient for "
                f"{', '.join(sorted(shortage_words))}; scanning only "
                "those target words in the corpus...",
                flush=True,
            )
            shortage_samples = [
                sample for sample in samples if sample.word in shortage_words
            ]
            context_token_counts = {
                sample.context_token_count for sample in shortage_samples
            }
            if len(context_token_counts) != 1:
                raise ValueError(
                    "Shortage samples do not share one context-token count"
                )
            fallback_occurrences, _ = index_corpus_occurrences(
                words=shortage_words,
                corpus_files=corpus_files,
                context_tokens=context_token_counts.pop(),
                exclude_capitalized_matches=True,
                max_occurrences_per_word=args.fallback_candidates,
                sampling_seed=13,
            )
            fallback_decisions = {
                sample.id: decisions[sample.id] for sample in shortage_samples
            }
            (
                fallback_replacements,
                fallback_rows,
                shortages,
            ) = find_replacements(
                shortage_samples,
                fallback_decisions,
                fallback_occurrences,
                validator=validator,
                model=args.model,
                workers=args.workers,
                candidate_batch_size=args.candidate_batch_size,
            )
            replacements.update(fallback_replacements)
            for row in fallback_rows:
                row["stage"] = "targeted_corpus_candidate"
            candidate_rows.extend(fallback_rows)
    _write_jsonl(args.audit_log, [*audit_rows, *candidate_rows])
    if shortages:
        details = ", ".join(
            f"{word}: need {needed}, found {found}"
            for word, (needed, found) in sorted(shortages.items())
        )
        print(
            f"Could not create a complete repaired dataset ({details}). "
            f"Audit log written to {args.audit_log}; output was not modified.",
            file=sys.stderr,
        )
        return 1
    repaired = apply_replacements(samples, decisions, replacements)
    _write_jsonl(args.output, repaired)
    print(
        f"Wrote {len(repaired)} contexts to {args.output}; "
        f"replaced {len(replacements)} and retained {len(repaired) - len(replacements)}.",
        flush=True,
    )
    print(f"Wrote all Gemini decisions to {args.audit_log}.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
