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
from wcs.audit import (
    AUDIT_SCHEMA_VERSION,
    DEFAULT_RANK_NEIGHBORS,
    DEFAULT_TOP_K,
    parse_temperature_list,
    temperature_slug,
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
    ModelSpec("gemma3-27b-base", "google/gemma-3-27b-pt", "gemma3", "base"),
    ModelSpec("gemma3-27b-it", "google/gemma-3-27b-it", "gemma3", "instruct"),
    ModelSpec("gemma4-e4b-base", "google/gemma-4-E4B", "gemma4", "base"),
    ModelSpec("gemma4-e4b-it", "google/gemma-4-E4B-it", "gemma4", "instruct"),
    ModelSpec("gemma2-9b-base", "google/gemma-2-9b", "gemma2", "base"),
    ModelSpec("gemma2-9b-it", "google/gemma-2-9b-it", "gemma2", "instruct"),
    ModelSpec("deepseek-v2-lite", "deepseek-ai/DeepSeek-V2-Lite", "deepseek-v2", "base"),
    ModelSpec(
        "deepseek-qwen14b-distill",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        "deepseek",
        "distill",
    ),
]

ENGLISH_PG19_MODEL_SLUGS = (
    "llama31-8b-base",
    "llama31-8b-instruct",
    "mistral7b-v03-base",
    "mistral7b-v03-instruct",
    "qwen35-9b-base",
    "qwen35-9b-instruct",
    "qwen25-14b-base",
    "qwen25-14b-instruct",
    "gemma3-12b-base",
    "gemma3-12b-it",
    "gemma3-27b-base",
    "gemma3-27b-it",
    "gemma4-e4b-base",
    "gemma4-e4b-it",
    "gemma2-9b-base",
    "gemma2-9b-it",
    "deepseek-qwen14b-distill",
)

ENGLISH_PG19_A100_MODEL_SLUGS = tuple(
    slug
    for slug in ENGLISH_PG19_MODEL_SLUGS
    if slug not in {"gemma3-27b-base", "gemma3-27b-it"}
)


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


def audit_is_complete(
    path: Path,
    expected_samples: int,
    *,
    required_top_k: int = DEFAULT_TOP_K,
    required_rank_neighbors: int = DEFAULT_RANK_NEIGHBORS,
) -> bool:
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
                if int(row.get("audit_schema_version", 0)) < AUDIT_SCHEMA_VERSION:
                    return False
                if len(row.get("top_5_tokens") or []) < required_top_k:
                    return False
                if len(row.get("top_5_probs") or []) < required_top_k:
                    return False
                if int(row.get("rank_neighbor_count", -1)) != required_rank_neighbors:
                    return False
                if not isinstance(row.get("rank_neighbors_above"), list):
                    return False
                if not isinstance(row.get("rank_neighbors_below"), list):
                    return False
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
    if raw.strip() == "english-pg19":
        wanted = set(ENGLISH_PG19_MODEL_SLUGS)
        return [model for model in DEFAULT_MODELS if model.slug in wanted]
    if raw.strip() == "english-pg19-a100":
        wanted = set(ENGLISH_PG19_A100_MODEL_SLUGS)
        return [model for model in DEFAULT_MODELS if model.slug in wanted]
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

    if audit_is_complete(
        output_path,
        expected_samples,
        required_top_k=args.top_k,
        required_rank_neighbors=args.rank_neighbors,
    ):
        print(f"[skip] {model.slug}: complete output already exists at {output_path}", flush=True)
        return output_path
    for alias in model.aliases:
        alias_path = args.results_dir / alias
        if audit_is_complete(
            alias_path,
            expected_samples,
            required_top_k=args.top_k,
            required_rank_neighbors=args.rank_neighbors,
        ):
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
        "--temperature",
        str(args.temperature),
        "--top-k",
        str(args.top_k),
        "--rank-neighbors",
        str(args.rank_neighbors),
        "--resume",
    ]
    if args.device_map:
        command.extend(["--device-map", args.device_map])
    if args.max_memory:
        command.extend(["--max-memory", args.max_memory])
    if args.offload_folder:
        command.extend(["--offload-folder", str(args.offload_folder)])
    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])
    if args.trust_remote_code:
        command.append("--trust-remote-code")

    quoted = " ".join(shlex.quote(part) for part in command)
    for attempt in range(1, args.retries + 2):
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

        if result.returncode == 0 and audit_is_complete(
            partial_path,
            expected_samples,
            required_top_k=args.top_k,
            required_rank_neighbors=args.rank_neighbors,
        ):
            partial_path.replace(output_path)
            print(f"[done] {model.slug}: wrote {output_path}", flush=True)
            return output_path

        print(
            f"[fail] {model.slug}: return code {result.returncode}; see {log_path}",
            flush=True,
        )
        print_log_tail(log_path)
        if STOP_REQUESTED:
            return None
        if attempt <= args.retries:
            time.sleep(args.retry_sleep_seconds)

    return None


def run_one_model_temperatures(
    model: ModelSpec,
    args: argparse.Namespace,
    expected_samples: int,
    env: dict[str, str],
    temperatures: list[float],
) -> dict[float, Path] | None:
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.logs_dir.mkdir(parents=True, exist_ok=True)

    output_paths = {
        temperature: args.results_dir / temperature_slug(temperature) / f"audit.{model.slug}.jsonl"
        for temperature in temperatures
    }
    partial_paths = {
        temperature: args.results_dir / temperature_slug(temperature) / f"audit.{model.slug}.jsonl.partial"
        for temperature in temperatures
    }
    for temperature, output_path in output_paths.items():
        if not audit_is_complete(
            output_path,
            expected_samples,
            required_top_k=args.top_k,
            required_rank_neighbors=args.rank_neighbors,
        ):
            break
    else:
        print(f"[skip] {model.slug}: complete outputs already exist for all temperatures", flush=True)
        return output_paths

    log_path = args.logs_dir / f"audit.{model.slug}.multi_temperature.log"
    output_template = str(args.results_dir / "{temperature_slug}" / f"audit.{model.slug}.jsonl.partial")
    command = [
        str(args.python),
        "-u",
        str(ROOT / "scripts" / "run_audit.py"),
        "--samples",
        str(args.samples),
        "--output",
        str(partial_paths[temperatures[0]]),
        "--output-template",
        output_template,
        "--model",
        model.model_id,
        "--device",
        args.device,
        "--dtype",
        args.dtype,
        "--temperatures",
        ",".join(f"{temperature:g}" for temperature in temperatures),
        "--top-k",
        str(args.top_k),
        "--rank-neighbors",
        str(args.rank_neighbors),
        "--resume",
    ]
    if args.device_map:
        command.extend(["--device-map", args.device_map])
    if args.max_memory:
        command.extend(["--max-memory", args.max_memory])
    if args.offload_folder:
        command.extend(["--offload-folder", str(args.offload_folder)])
    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])
    if args.trust_remote_code:
        command.append("--trust-remote-code")

    quoted = " ".join(shlex.quote(part) for part in command)
    for attempt in range(1, args.retries + 2):
        started = time.strftime("%Y-%m-%d %H:%M:%S")
        print(
            f"[run] {model.slug} multi-temperature attempt {attempt}: {model.model_id}",
            flush=True,
        )
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"\n===== {started} {model.slug} multi-temperature attempt {attempt} =====\n")
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

        complete = result.returncode == 0 and all(
            audit_is_complete(
                path,
                expected_samples,
                required_top_k=args.top_k,
                required_rank_neighbors=args.rank_neighbors,
            )
            for path in partial_paths.values()
        )
        if complete:
            for temperature, partial_path in partial_paths.items():
                output_path = output_paths[temperature]
                output_path.parent.mkdir(parents=True, exist_ok=True)
                partial_path.replace(output_path)
            print(f"[done] {model.slug}: wrote {len(temperatures)} temperature outputs", flush=True)
            return output_paths

        print(
            f"[fail] {model.slug}: return code {result.returncode}; see {log_path}",
            flush=True,
        )
        print_log_tail(log_path)
        if STOP_REQUESTED:
            return None
        if attempt <= args.retries:
            time.sleep(args.retry_sleep_seconds)

    return None


def print_log_tail(path: Path, lines: int = 80) -> None:
    try:
        rows = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return
    print(f"[log-tail] last {min(lines, len(rows))} lines from {path}", flush=True)
    for row in rows[-lines:]:
        print(row, flush=True)


def write_summaries(paths: list[Path], summary_path: Path, word_summary_path: Path) -> None:
    if not paths:
        return
    rows = summarize_wcs(paths)
    write_summary_csv(rows, summary_path)
    print(f"[summary] wrote {summary_path}", flush=True)
    word_rows = summarize_wcs_by_target_word(paths)
    write_word_summary_csv(word_rows, word_summary_path)
    print(f"[word-summary] wrote {word_summary_path}", flush=True)


def write_temperature_summaries(completed: dict[float, dict[str, Path]], args: argparse.Namespace) -> None:
    all_paths = []
    for temperature, by_slug in completed.items():
        all_paths.extend(by_slug.values())
    if not all_paths:
        return
    write_summaries(
        all_paths,
        args.summary,
        args.word_summary,
    )


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
    parser.add_argument("--device-map", default=None, help="Optional Transformers device_map, e.g. auto.")
    parser.add_argument(
        "--max-memory",
        default=None,
        help="Optional comma list for device_map, e.g. 0=44GiB,1=44GiB,cpu=160GiB.",
    )
    parser.add_argument("--offload-folder", type=Path, default=None, help="Optional folder for unquantized CPU/disk offload.")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--temperatures", default=None, help="Comma-separated temperatures for one-pass multi-temperature audits.")
    parser.add_argument(
        "--index-only",
        action="store_true",
        help=(
            "Validate and summarize existing multi-temperature audits without "
            "loading models or running inference. Requires --temperatures."
        ),
    )
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--rank-neighbors", type=int, default=DEFAULT_RANK_NEIGHBORS)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--models",
        default=None,
        help=(
            "Comma-separated model slugs/Hugging Face IDs, or the preset "
            "'english-pg19' (17 original models excluding Nemotron) or "
            "'english-pg19-a100' (also excludes both 27B models)."
        ),
    )
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--retry-sleep-seconds", type=int, default=60)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def main() -> int:
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    args = parse_args()
    if args.top_k < 0:
        raise SystemExit("--top-k must be non-negative")
    if args.rank_neighbors < 0:
        raise SystemExit("--rank-neighbors must be non-negative")
    
    # If the summary arguments are exactly their defaults, override them to live inside results_dir
    default_summary = ROOT / "results/wcs_summary.model_suite.csv"
    default_word_summary = ROOT / "results/wcs_word_summary.model_suite.csv"
    default_manifest = ROOT / "results/model_suite_manifest.json"
    
    if args.summary == default_summary:
        args.summary = args.results_dir / "wcs_summary.csv"
    if args.word_summary == default_word_summary:
        args.word_summary = args.results_dir / "wcs_word_summary.csv"
    if args.manifest == default_manifest:
        args.manifest = args.results_dir / "manifest.json"

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
    if args.index_only and not args.temperatures:
        raise SystemExit("--index-only requires --temperatures")
    if args.temperatures:
        temperatures = parse_temperature_list(args.temperatures)
        print(
            "[start] multi-temperature mode: "
            + ", ".join(f"T={temperature:g}" for temperature in temperatures),
            flush=True,
        )
        completed_by_temperature: dict[float, dict[str, Path]] = {
            temperature: {} for temperature in temperatures
        }
        if args.index_only:
            missing: list[str] = []
            for temperature in temperatures:
                for model in models:
                    path = (
                        args.results_dir
                        / temperature_slug(temperature)
                        / f"audit.{model.slug}.jsonl"
                    )
                    if audit_is_complete(
                        path,
                        expected_samples,
                        required_top_k=args.top_k,
                        required_rank_neighbors=args.rank_neighbors,
                    ):
                        completed_by_temperature[temperature][model.slug] = path
                        print(
                            f"[index] T={temperature:g} {model.slug}: complete",
                            flush=True,
                        )
                    else:
                        missing.append(f"T={temperature:g}:{model.slug}")
                        print(
                            f"[missing] T={temperature:g} {model.slug}: {path}",
                            flush=True,
                        )
            write_temperature_summaries(completed_by_temperature, args)
            if missing:
                print(
                    "[index-failed] incomplete audits: " + ", ".join(missing),
                    flush=True,
                )
                return 1
            print("[indexed] all requested existing audits are complete.", flush=True)
            return 0

        failures: list[str] = []
        for model in models:
            if STOP_REQUESTED:
                break
            output_paths = run_one_model_temperatures(
                model,
                args,
                expected_samples,
                env,
                temperatures,
            )
            if output_paths is None:
                failures.append(model.slug)
                continue
            for temperature, path in output_paths.items():
                completed_by_temperature[temperature][model.slug] = path
            write_temperature_summaries(completed_by_temperature, args)
        write_temperature_summaries(completed_by_temperature, args)
        if failures:
            print(f"[finished-with-failures] {', '.join(failures)}", flush=True)
            return 1
        if STOP_REQUESTED:
            print("[stopped] resume by running the same command again.", flush=True)
            return 130
        print("[finished] all selected models completed.", flush=True)
        return 0

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
