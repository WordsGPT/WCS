#!/usr/bin/env python3
"""Build compact, per-word data files for the FineWeb context browser."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any


TEMPERATURES = (0.6, 0.7, 1.0, 1.5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--audits", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def expand_paths(patterns: list[str]) -> list[Path]:
    paths: set[Path] = set()
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            paths.update(Path(match).resolve() for match in matches)
        elif Path(pattern).is_file():
            paths.add(Path(pattern).resolve())
    if not paths:
        raise SystemExit("No audit files matched --audits")
    return sorted(paths)


def read_audit_identity(path: Path) -> tuple[str, float]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                return str(row["model"]), float(row.get("temperature", 1.0))
    raise ValueError(f"Audit is empty: {path}")


def load_samples(path: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    words: dict[str, dict[str, Any]] = {}
    samples: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = str(row.get("id", row.get("sample_id", "")))
            word = str(row.get("word", row.get("matched_text", "")))
            if not sample_id or not word:
                raise ValueError(f"Sample is missing id or word in {path}")
            context = {
                "id": sample_id,
                "prefix": str(row.get("prefix", "")),
                "source": str(row.get("source_path", "")),
                "results": {},
            }
            word_data = words.setdefault(
                word,
                {
                    "word": word,
                    "rank": row.get("rank"),
                    "count": row.get("count"),
                    "contexts": [],
                },
            )
            word_data["contexts"].append(context)
            samples[sample_id] = context
    ordered = sorted(words.values(), key=lambda item: (str(item["word"]).lower(), str(item["word"])))
    return ordered, samples


def compact_float(value: float) -> float:
    return float(f"{value:.8g}")


def main() -> None:
    args = parse_args()
    audit_paths = expand_paths(args.audits)
    words, samples = load_samples(args.samples)

    identities = {path: read_audit_identity(path) for path in audit_paths}
    models = sorted({model for model, _temperature in identities.values()})
    temperatures = sorted({temperature for _model, temperature in identities.values()})
    if temperatures != list(TEMPERATURES):
        raise ValueError(f"Expected temperatures {TEMPERATURES}, found {temperatures}")
    if len(models) != 13:
        raise ValueError(f"Expected 13 models, found {len(models)}")

    model_index = {model: index for index, model in enumerate(models)}
    temperature_index = {temperature: index for index, temperature in enumerate(temperatures)}

    for number, audit_path in enumerate(audit_paths, start=1):
        model, temperature = identities[audit_path]
        key = f"{temperature_index[temperature]}:{model_index[model]}"
        print(f"[{number:02d}/{len(audit_paths)}] {model} at T={temperature:g}", flush=True)
        with audit_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                context = samples.get(str(row.get("sample_id", "")))
                if context is None:
                    continue
                rank = int(row["rank"])
                mass_above = float(row["cumulative_probability"]) - float(row["probability"])
                ratio = float(row["probability_ratio_to_top"])
                probability = float(row["probability"])
                current = context["results"].get(key)
                if current is None:
                    # worst rank, largest mass above, smallest min-p ratio,
                    # forced-path token probability product, token count
                    context["results"][key] = [
                        rank,
                        compact_float(mass_above),
                        compact_float(ratio),
                        compact_float(probability),
                        1,
                    ]
                else:
                    current[0] = max(current[0], rank)
                    current[1] = compact_float(max(current[1], mass_above))
                    current[2] = compact_float(min(current[2], ratio))
                    current[3] = compact_float(current[3] * probability)
                    current[4] += 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    index_words = []
    for index, word_data in enumerate(words):
        filename = f"word-{index:03d}.json"
        contexts = word_data["contexts"]
        missing = [
            context["id"]
            for context in contexts
            if len(context["results"]) != len(models) * len(temperatures)
        ]
        if missing:
            raise ValueError(
                f"{word_data['word']!r} has {len(missing)} contexts with incomplete results"
            )
        payload = {
            "word": word_data["word"],
            "rank": word_data["rank"],
            "count": word_data["count"],
            "contexts": contexts,
        }
        with (args.output_dir / filename).open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
        index_words.append(
            {
                "word": word_data["word"],
                "rank": word_data["rank"],
                "count": word_data["count"],
                "file": filename,
                "contexts": len(contexts),
            }
        )

    index_payload = {
        "metadata": {
            "dataset": "HuggingFaceFW/fineweb",
            "config": "sample-10BT",
            "models": len(models),
            "temperatures": temperatures,
            "contexts": len(samples),
        },
        "models": models,
        "temperatures": temperatures,
        "words": index_words,
    }
    with (args.output_dir / "index.json").open("w", encoding="utf-8") as handle:
        json.dump(index_payload, handle, ensure_ascii=False, separators=(",", ":"))
    total_mib = sum(path.stat().st_size for path in args.output_dir.glob("*.json")) / 1024 / 1024
    print(
        f"Wrote {len(index_words)} word files for {len(samples):,} contexts "
        f"and {len(models)} models ({total_mib:.1f} MiB)",
        flush=True,
    )


if __name__ == "__main__":
    main()
