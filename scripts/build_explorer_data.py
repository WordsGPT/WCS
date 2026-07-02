#!/usr/bin/env python
"""Build the static context-explorer payload from samples and audit logs."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument(
        "--audits",
        nargs="+",
        required=True,
        help="Audit JSONL paths or glob patterns.",
    )
    parser.add_argument("--output", type=Path, default=Path("explorer_data.json"))
    parser.add_argument("--dataset", default="WCS")
    parser.add_argument("--language", default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--allow-missing-predictions",
        action="store_true",
        help="Allow legacy audits without top-k and rank-neighbor fields.",
    )
    return parser.parse_args()


def expand_paths(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = [Path(match) for match in glob.glob(pattern)]
        if matches:
            paths.extend(matches)
        elif Path(pattern).is_file():
            paths.append(Path(pattern))
    unique = sorted({path.resolve() for path in paths})
    if not unique:
        raise SystemExit("No audit files matched --audits")
    return unique


def normalize_predictions(rows: Any) -> list[dict[str, Any]]:
    if not isinstance(rows, list):
        return []
    normalized = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        normalized.append(
            {
                "rank": int(row["rank"]),
                "id": int(row["token_id"]),
                "t": str(row["token_text"]),
                "p": float(row["probability"]),
            }
        )
    return normalized


def load_samples(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    data: dict[str, Any] = {"models": [], "words": {}}
    contexts: dict[str, dict[str, Any]] = {}

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            raw_sample_id = row.get("id", row.get("sample_id"))
            raw_word = row.get("word", row.get("matched_text"))
            if raw_sample_id is None or raw_word is None:
                raise ValueError(f"Sample row is missing id/word in {path}")
            sample_id = str(raw_sample_id)
            word = str(raw_word)
            if not sample_id or not word:
                raise ValueError(f"Sample row has an empty id/word in {path}")
            if sample_id in contexts:
                raise ValueError(f"Duplicate sample id {sample_id!r} in {path}")

            word_data = data["words"].setdefault(
                word,
                {"word": word, "rank": row.get("rank"), "contexts": []},
            )
            context = {
                "id": sample_id,
                "prefix": row.get("prefix", ""),
                "target": word,
                "results": {},
            }
            word_data["contexts"].append(context)
            contexts[sample_id] = context

    return data, contexts


def main() -> None:
    args = parse_args()
    audit_paths = expand_paths(args.audits)
    data, contexts = load_samples(args.samples)
    data["metadata"] = {
        "dataset": args.dataset,
        "language": args.language,
        "temperature": args.temperature,
        "samples": len(contexts),
    }

    models: set[str] = set()
    seen_steps: set[tuple[str, str, int]] = set()

    for audit_path in audit_paths:
        print(f"Processing {audit_path}", flush=True)
        with audit_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                sample_id = str(row.get("sample_id", ""))
                if sample_id not in contexts:
                    continue
                temperature = float(row.get("temperature", 1.0))
                if abs(temperature - args.temperature) > 1e-9:
                    continue

                model = str(row.get("model", ""))
                if not model:
                    raise ValueError(f"Audit row in {audit_path} is missing model")
                word_token_index = int(row.get("word_token_index", 0))
                step_key = (sample_id, model, word_token_index)
                if step_key in seen_steps:
                    raise ValueError(
                        "Duplicate audit step for "
                        f"sample={sample_id}, model={model}, token={word_token_index}"
                    )
                seen_steps.add(step_key)
                models.add(model)

                result = contexts[sample_id]["results"].setdefault(
                    model,
                    {
                        "rank": 0,
                        "prob": 1.0,
                        "tokenSteps": [],
                        "top5": [],
                        "neighborsAbove": [],
                        "neighborsBelow": [],
                    },
                )
                rank = int(row.get("rank", 0))
                probability = float(row.get("probability", 0.0))
                top_tokens = row.get("top_5_tokens") or []
                top_probs = row.get("top_5_probs") or []
                top_predictions = [
                    {"rank": index + 1, "t": str(token), "p": float(prob)}
                    for index, (token, prob) in enumerate(zip(top_tokens, top_probs))
                ]
                neighbors_above = normalize_predictions(
                    row.get("rank_neighbors_above")
                )
                neighbors_below = normalize_predictions(
                    row.get("rank_neighbors_below")
                )
                if not args.allow_missing_predictions and (
                    not top_predictions
                    or "rank_neighbors_above" not in row
                    or "rank_neighbors_below" not in row
                ):
                    raise ValueError(
                        f"{audit_path} is a legacy audit without prediction details; "
                        "rerun it with the current auditor"
                    )

                result["prob"] *= probability
                result["rank"] = max(result["rank"], rank)
                result["tokenSteps"].append(
                    {
                        "index": word_token_index,
                        "id": int(row.get("token_id", -1)),
                        "t": str(row.get("token_text", "")),
                        "rank": rank,
                        "p": probability,
                        "top5": top_predictions,
                        "neighborsAbove": neighbors_above,
                        "neighborsBelow": neighbors_below,
                    }
                )

                if word_token_index != 0:
                    continue

                result["top5"] = top_predictions
                result["neighborsAbove"] = neighbors_above
                result["neighborsBelow"] = neighbors_below
                result["targetToken"] = {
                    "rank": rank,
                    "id": int(row.get("token_id", -1)),
                    "t": str(row.get("token_text", "")),
                    "p": probability,
                }

    data["models"] = sorted(models)
    for word_data in data["words"].values():
        for context in word_data["contexts"]:
            for result in context["results"].values():
                result["tokenSteps"].sort(key=lambda step: step["index"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, separators=(",", ":"))
    size_mib = args.output.stat().st_size / 1024 / 1024
    print(
        f"Wrote {args.output} with {len(data['models'])} models and "
        f"{len(contexts)} contexts ({size_mib:.2f} MiB)",
        flush=True,
    )


if __name__ == "__main__":
    main()
