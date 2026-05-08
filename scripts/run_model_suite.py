#!/usr/bin/env python
"""Resumable WCS audit runner for the paper model suite."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from wcs.metrics import (
    summarize_wcs,
    summarize_wcs_by_target_word,
    write_summary_csv,
    write_word_summary_csv,
)


@dataclass(frozen=True)
class ModelSpec:
    slug: str
    model_id: str
    family: str
    variant: str
    aliases: tuple[str, ...] = ()


DEFAULT_MODELS = [
    ModelSpec("llama31-8b-base", "meta-llama/Llama-3.1-8B", "llama", "base"),
    ModelSpec(
        "llama31-8b-instruct",
        "meta-llama/Llama-3.1-8B-Instruct",
        "llama",
        "instruct",
    ),
    ModelSpec(
        "mistral7b-v03-base",
        "mistralai/Mistral-7B-v0.3",
        "mistral",
        "base",
        aliases=("audit.mistral7b.limit1000.jsonl",),
    ),
    ModelSpec(
        "mistral7b-v03-instruct",
        "mistralai/Mistral-7B-Instruct-v0.3",
        "mistral",
        "instruct",
        aliases=("audit.mistral7b-instruct.full.jsonl",),
    ),
    ModelSpec("qwen35-9b-base", "Qwen/Qwen3.5-9B-Base", "qwen-small", "base"),
    ModelSpec(
        "qwen35-9b-instruct",
        "Qwen/Qwen3.5-9B",
        "qwen-small",
        "instruct",
    ),
    ModelSpec("qwen25-14b-base", "Qwen/Qwen2.5-14B", "qwen-mid", "base"),
    ModelSpec(
        "qwen25-14b-instruct",
        "Qwen/Qwen2.5-14B-Instruct",
        "qwen-mid",
        "instruct",
    ),
    ModelSpec("gemma3-12b-base", "google/gemma-3-12b-pt", "gemma3", "base"),
    ModelSpec("gemma3-12b-it", "google/gemma-3-12b-it", "gemma3", "instruct"),
    ModelSpec("gemma4-e4b-base", "google/gemma-4-E4B", "gemma4", "base"),
    ModelSpec("gemma4-e4b-it", "google/gemma-4-E4B-it", "gemma4", "instruct"),
    ModelSpec(
        "deepseek-qwen14b-distill",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        "deepseek",
        "distill",
    ),
]


STOP_REQUESTED = False


def request_stop(signum: int, _frame: object) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print(f"\nReceived signal {signum}; stopping after the current model.", flush=True)


def sample_count(path: Path, limit: int | None) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
                if limit is not None and count >= limit:
                    break
    return count


def audit_is_complete(path: Path, expected_samples: int) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False

    sample_ids: set[str] = set()
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                sample_ids.add(str(row["sample_id"]))
    except (OSError, json.JSONDecodeError, KeyError):
        return False

    return len(sample_ids) == expected_samples


def write_manifest(path: Path, completed: dict[str, Path], models: Iterable[ModelSpec]) -> None:
    rows = []
    for model in models:
        output_path = completed.get(model.slug)
        rows.append(
            {
                "slug": model.slug,
                "model_id": model.model_id,
                "family": model.family,
                "variant": model.variant,
                "audit_path": str(output_path) if output_path else None,
                "completed": output_path is not None,
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


def select_models(raw: str | None) -> list[ModelSpec]:
    if not raw:
        return list(DEFAULT_MODELS)
    wanted = {part.strip() for part in raw.split(",") if part.strip()}
    selected = [model for model in DEFAULT_MODELS if model.slug in wanted or model.model_id in wanted]
    missing = sorted(wanted - {model.slug for model in selected} - {model.model_id for model in selected})
    if missing:
        raise SystemExit(f"Unknown model slug/id in --models: {', '.join(missing)}")
    return selected


def run_one_model(
    model: ModelSpec,
    args: argparse.Namespace,
    expected_samples: int,
    env: dict[str, str],
) -> Path | None:
    output_path = args.results_dir / f"audit.{model.slug}.jsonl"
    partial_path = args.results_dir / f"audit.{model.slug}.jsonl.partial"
    log_path = args.logs_dir / f"audit.{model.slug}.log"

    if audit_is_complete(output_path, expected_samples):
        print(f"[skip] {model.slug}: complete output already exists at {output_path}", flush=True)
        return output_path
    for alias in model.aliases:
        alias_path = args.results_dir / alias
        if audit_is_complete(alias_path, expected_samples):
            print(
                f"[skip] {model.slug}: using complete existing output at {alias_path}",
                flush=True,
            )
            return alias_path

    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.logs_dir.mkdir(parents=True, exist_ok=True)

    command = [
        str(args.python),
        "-u",
        str(ROOT / "scripts" / "run_audit.py"),
        "--samples",
        str(args.samples),
        "--output",
        str(partial_path),
        "--model",
        model.model_id,
        "--device",
        args.device,
        "--dtype",
        args.dtype,
    ]
    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])
    if args.trust_remote_code:
        command.append("--trust-remote-code")

    quoted = " ".join(shlex.quote(part) for part in command)
    for attempt in range(1, args.retries + 2):
        if partial_path.exists():
            partial_path.unlink()

        started = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[run] {model.slug} attempt {attempt}: {model.model_id}", flush=True)
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"\n===== {started} {model.slug} attempt {attempt} =====\n")
            log.write(quoted + "\n")
            log.flush()
            result = subprocess.run(
                command,
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )

        if result.returncode == 0 and audit_is_complete(partial_path, expected_samples):
            partial_path.replace(output_path)
            print(f"[done] {model.slug}: wrote {output_path}", flush=True)
            return output_path

        print(
            f"[fail] {model.slug}: return code {result.returncode}; see {log_path}",
            flush=True,
        )
        if STOP_REQUESTED:
            return None
        if attempt <= args.retries:
            time.sleep(args.retry_sleep_seconds)

    return None


def write_summaries(paths: list[Path], summary_path: Path, word_summary_path: Path) -> None:
    if not paths:
        return
    rows = summarize_wcs(paths)
    write_summary_csv(rows, summary_path)
    print(f"[summary] wrote {summary_path}", flush=True)
    word_rows = summarize_wcs_by_target_word(paths)
    write_word_summary_csv(word_rows, word_summary_path)
    print(f"[word-summary] wrote {word_summary_path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the WCS model suite with resume/retry support.")
    parser.add_argument("--samples", type=Path, default=ROOT / "data/processed/samples.jsonl")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--logs-dir", type=Path, default=ROOT / "logs/model_suite")
    parser.add_argument("--summary", type=Path, default=ROOT / "results/wcs_summary.model_suite.csv")
    parser.add_argument(
        "--word-summary",
        type=Path,
        default=ROOT / "results/wcs_word_summary.model_suite.csv",
    )
    parser.add_argument("--manifest", type=Path, default=ROOT / "results/model_suite_manifest.json")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--models", default=None, help="Comma-separated model slugs or Hugging Face IDs.")
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--retry-sleep-seconds", type=int, default=60)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def main() -> int:
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    args = parse_args()
    args.samples = args.samples.resolve()
    args.results_dir = args.results_dir.resolve()
    args.logs_dir = args.logs_dir.resolve()
    args.summary = args.summary.resolve()
    args.word_summary = args.word_summary.resolve()
    args.manifest = args.manifest.resolve()

    models = select_models(args.models)
    expected_samples = sample_count(args.samples, args.limit)
    env = os.environ.copy()
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("MPLBACKEND", "Agg")

    print(f"[start] {len(models)} models; expected samples per model: {expected_samples}", flush=True)
    completed: dict[str, Path] = {}
    failures: list[str] = []

    for model in models:
        if STOP_REQUESTED:
            break
        output_path = run_one_model(model, args, expected_samples, env)
        if output_path is None:
            failures.append(model.slug)
            continue
        completed[model.slug] = output_path
        write_manifest(args.manifest, completed, models)
        write_summaries(list(completed.values()), args.summary, args.word_summary)

    write_manifest(args.manifest, completed, models)
    write_summaries(list(completed.values()), args.summary, args.word_summary)

    if failures:
        print(f"[finished-with-failures] {', '.join(failures)}", flush=True)
        return 1
    if STOP_REQUESTED:
        print("[stopped] resume by running the same command again.", flush=True)
        return 130
    print("[finished] all selected models completed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
