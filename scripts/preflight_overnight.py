#!/usr/bin/env python
"""Fail-fast checks for credentials, FineWeb, model access, GPU, disk, and git."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_model_suite import select_models
from wcs.dataset_builder import is_text_coherent, load_env_file


def check(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)
    print(f"[ok] {message}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", required=True)
    parser.add_argument("--dataset", default="HuggingFaceFW/fineweb")
    parser.add_argument("--config", default="sample-10BT")
    parser.add_argument("--min-free-disk-gib", type=float, default=140.0)
    parser.add_argument("--skip-gemini", action="store_true")
    parser.add_argument("--skip-git", action="store_true")
    args = parser.parse_args()

    import shutil
    import torch
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download, model_info

    load_env_file(ROOT / ".env")
    check(torch.cuda.is_available(), "CUDA is available")
    props = torch.cuda.get_device_properties(0)
    print(f"[gpu] {props.name}; {props.total_memory / 2**30:.1f} GiB; torch CUDA {torch.version.cuda}")
    check(props.total_memory >= 40 * 2**30, "GPU has at least 40 GiB VRAM")
    free_gib = shutil.disk_usage(ROOT).free / 2**30
    check(free_gib >= args.min_free_disk_gib, f"free disk is {free_gib:.1f} GiB")

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    for model in select_models(args.models):
        info = model_info(model.model_id, token=token)
        check(info.id is not None, f"Hugging Face access: {model.model_id}")
        hf_hub_download(model.model_id, "config.json", token=token)
        print(f"[ok] downloaded config: {model.model_id}", flush=True)

    stream = load_dataset(
        args.dataset,
        name=args.config,
        split="train",
        streaming=True,
        columns=["text", "id", "url", "dump", "language_score", "token_count"],
    )
    first = next(iter(stream))
    check(bool(first.get("text")), f"streamed one {args.dataset}/{args.config} document")

    if not args.skip_gemini:
        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        check(bool(key), "Gemini API key is set")
        coherent = is_text_coherent(
            "This is a short, coherent English sentence ending with the target word experiment",
            target_word="experiment",
        )
        check(type(coherent) is bool, "Gemini structured coherence request succeeded")

    if not args.skip_git:
        result = subprocess.run(
            ["git", "ls-remote", "--exit-code", "origin", "HEAD"],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
            check=False,
        )
        check(result.returncode == 0, f"git remote is reachable ({result.stderr.strip() or 'origin'})")
    print("[preflight] all checks passed", flush=True)


if __name__ == "__main__":
    main()
