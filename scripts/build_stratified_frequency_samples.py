#!/usr/bin/env python
"""Build a WCS sample set from multiple frequency-rank strata."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from wcs.dataset_builder import DEFAULT_GEMINI_MODEL, Sample, build_samples


DEFAULT_STRATA = "1:1000:200,1001:10000:400,10001:100000:400"


def parse_strata(raw: str) -> list[tuple[int, int, int]]:
    strata: list[tuple[int, int, int]] = []
    for part in raw.split(","):
        cleaned = part.strip()
        if not cleaned:
            continue
        fields = cleaned.split(":")
        if len(fields) != 3:
            raise ValueError("Each stratum must look like rank_min:rank_max:word_count")
        rank_min, rank_max, word_count = (int(field) for field in fields)
        if rank_min < 1 or rank_max < rank_min or word_count < 1:
            raise ValueError(f"Invalid stratum: {cleaned}")
        strata.append((rank_min, rank_max, word_count))
    if not strata:
        raise ValueError("At least one stratum is required")
    return strata


def checkpoint_path(work_dir: Path, rank_min: int, rank_max: int, word_count: int) -> Path:
    return work_dir / f"samples.r{rank_min}-{rank_max}.n{word_count}.jsonl"


def rewrite_sample(
    sample: Sample,
    sample_id: int,
    rank_min: int,
    rank_max: int,
    word_count: int,
) -> Sample:
    metadata = dict(sample.metadata)
    metadata.update(
        {
            "stratum": f"{rank_min}-{rank_max}",
            "stratum_rank_min": rank_min,
            "stratum_rank_max": rank_max,
            "stratum_target_words": word_count,
            "experiment": "frequency_wcs",
        }
    )
    return replace(sample, id=f"sample-{sample_id:06d}", metadata=metadata)


def write_jsonl(samples: list[Sample], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(asdict(sample), ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frequency", type=Path, default=ROOT / "data/raw/norvig_count_1w.txt")
    parser.add_argument("--corpus", type=Path, required=True, help="Corpus file or directory of .txt/.text files.")
    parser.add_argument("--output", type=Path, default=ROOT / "data/processed/samples.frequency_1k.jsonl")
    parser.add_argument("--work-dir", type=Path, default=None, help="Per-stratum checkpoint directory.")
    parser.add_argument("--strata", default=DEFAULT_STRATA, help="Comma list rank_min:rank_max:word_count.")
    parser.add_argument("--context-tokens", type=int, default=256)
    parser.add_argument("--contexts-per-word", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260527)
    parser.add_argument("--min-word-length", type=int, default=3)
    parser.add_argument("--dictionary", type=Path, default=None)
    parser.add_argument("--exclude-capitalized-matches", action="store_true")
    parser.add_argument("--coherence-model", default=DEFAULT_GEMINI_MODEL)
    parser.add_argument("--skip-coherence-check", action="store_true")
    parser.add_argument("--language", default="English")
    parser.add_argument("--progress-interval", type=int, default=25)
    parser.add_argument("--resume", action="store_true", help="Resume per-stratum checkpoints if present.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    strata = parse_strata(args.strata)
    work_dir = args.work_dir or args.output.with_suffix(".checkpoints")
    work_dir.mkdir(parents=True, exist_ok=True)

    combined: list[Sample] = []
    all_missing = 0
    next_sample_id = 1
    for stratum_index, (rank_min, rank_max, word_count) in enumerate(strata, start=1):
        checkpoint = checkpoint_path(work_dir, rank_min, rank_max, word_count)
        print(
            f"[stratum {stratum_index}/{len(strata)}] ranks {rank_min}-{rank_max}; "
            f"target words={word_count}; checkpoint={checkpoint}",
            flush=True,
        )
        samples, missing = build_samples(
            frequency_path=args.frequency,
            corpus_path=args.corpus,
            rank_min=rank_min,
            rank_max=rank_max,
            sample_size=word_count,
            context_tokens=args.context_tokens,
            seed=args.seed + stratum_index - 1,
            exclude_capitalized_matches=args.exclude_capitalized_matches,
            min_word_length=args.min_word_length,
            dictionary_path=args.dictionary,
            contexts_per_word=args.contexts_per_word,
            coherence_model=args.coherence_model,
            language=args.language,
            checkpoint_path=checkpoint,
            progress_interval=args.progress_interval,
            resume=args.resume,
            skip_coherence_check=args.skip_coherence_check,
        )
        all_missing += len(missing)
        for sample in samples:
            combined.append(rewrite_sample(sample, next_sample_id, rank_min, rank_max, word_count))
            next_sample_id += 1
        write_jsonl(combined, args.output)
        accepted_words = len({sample.word for sample in samples})
        print(
            f"[stratum {stratum_index}/{len(strata)}] accepted {accepted_words} words / "
            f"{len(samples)} contexts",
            flush=True,
        )

    write_jsonl(combined, args.output)
    total_words = len({sample.word for sample in combined})
    print(f"Wrote {len(combined)} samples for {total_words} words to {args.output}")
    if all_missing:
        print(f"Skipped {all_missing} candidate words without enough usable contexts")


if __name__ == "__main__":
    main()
