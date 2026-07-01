#!/usr/bin/env python
"""Merge repaired-sample audit rows into complete existing audit files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_changed_ids(path: Path) -> set[str]:
    with path.open("r", encoding="utf-8") as handle:
        return {
            str(json.loads(line)["id"])
            for line in handle
            if line.strip()
        }


def load_audit_groups(path: Path) -> tuple[list[str], dict[str, list[dict]]]:
    order: list[str] = []
    groups: dict[str, list[dict]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = str(row["sample_id"])
            if sample_id not in groups:
                order.append(sample_id)
                groups[sample_id] = []
            groups[sample_id].append(row)
    return order, groups


def merge_audit_file(
    base_path: Path,
    delta_path: Path,
    output_path: Path,
    changed_ids: set[str],
) -> None:
    base_order, base = load_audit_groups(base_path)
    _delta_order, delta = load_audit_groups(delta_path)
    missing_base = changed_ids - base.keys()
    missing_delta = changed_ids - delta.keys()
    unexpected_delta = delta.keys() - changed_ids
    if missing_base or missing_delta or unexpected_delta:
        raise ValueError(
            f"Cannot merge {delta_path}: missing_base={sorted(missing_base)[:5]}, "
            f"missing_delta={sorted(missing_delta)[:5]}, "
            f"unexpected_delta={sorted(unexpected_delta)[:5]}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for sample_id in base_order:
            rows = delta[sample_id] if sample_id in changed_ids else base[sample_id]
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(output_path)


def merge_directories(
    base_dir: Path,
    delta_dir: Path,
    output_dir: Path,
    changed_samples_path: Path,
) -> int:
    changed_ids = load_changed_ids(changed_samples_path)
    if not changed_ids:
        raise ValueError(f"No changed samples found in {changed_samples_path}")
    delta_paths = sorted(delta_dir.rglob("audit.*.jsonl"))
    if not delta_paths:
        raise FileNotFoundError(f"No completed audit files found under {delta_dir}")
    for delta_path in delta_paths:
        relative = delta_path.relative_to(delta_dir)
        base_path = base_dir / relative
        if not base_path.exists():
            raise FileNotFoundError(f"Missing base audit corresponding to {relative}")
        merge_audit_file(
            base_path,
            delta_path,
            output_dir / relative,
            changed_ids,
        )
        print(f"Merged {relative}")
    return len(delta_paths)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replace changed sample rows in full audit files with delta results."
    )
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--delta-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--changed-samples", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = merge_directories(
        args.base_dir,
        args.delta_dir,
        args.output_dir,
        args.changed_samples,
    )
    print(f"Merged {count} audit files into {args.output_dir}")


if __name__ == "__main__":
    main()
