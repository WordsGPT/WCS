"""Forced-path model audit for Word Coverage Score.

The evaluator records how reachable each target-word token is under common
sampling filters. It does not sample text.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator


@dataclass(frozen=True)
class AuditTokenRow:
    sample_id: str
    model: str
    word: str
    word_rank: int
    source_path: str
    word_token_index: int
    token_id: int
    token_text: str
    rank: int
    probability: float
    top_probability: float
    probability_ratio_to_top: float
    cumulative_probability: float
    context_token_count: int
    prefix_char_count: int
    word_token_count: int


def load_samples(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            samples.append(json.loads(line))
            if limit is not None and len(samples) >= limit:
                break
    return samples


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def encode_target_word(tokenizer: Any, word: str) -> list[int]:
    """Encode a target as a continuation after whitespace.

    Most causal LMs tokenize the first word token differently when it begins
    after a space. The sample prefix does not include the target, so the audit
    treats the target word as the next whitespace-delimited continuation.
    """

    token_ids = tokenizer.encode(" " + word, add_special_tokens=False)
    if not token_ids:
        raise ValueError(f"Tokenizer produced no ids for target word {word!r}")
    return list(token_ids)


def encode_prefix(tokenizer: Any, prefix: str) -> list[int]:
    token_ids = tokenizer.encode(prefix, add_special_tokens=False)
    if not token_ids:
        raise ValueError("Tokenizer produced no ids for sample prefix")
    return list(token_ids)


def rank_probability_from_logits(logits: Any, token_id: int) -> tuple[int, float, float, float, float]:
    """Return rank/probability diagnostics for one token id.

    Returns:
        rank, target_probability, top_probability, ratio_to_top, cumulative_probability
    """

    import torch

    probs = torch.softmax(logits.float(), dim=-1)
    target_prob = probs[token_id]
    top_prob = torch.max(probs)
    rank = int(torch.count_nonzero(probs > target_prob).item() + 1)
    sorted_probs, _ = torch.sort(probs, descending=True)
    cumulative_probability = float(torch.sum(sorted_probs[:rank]).item())
    ratio = float((target_prob / top_prob).item()) if float(top_prob.item()) > 0 else 0.0
    return (
        rank,
        float(target_prob.item()),
        float(top_prob.item()),
        ratio,
        cumulative_probability,
    )


def audit_sample(
    model: Any,
    tokenizer: Any,
    sample: dict[str, Any],
    model_name: str,
    device: str | None = None,
) -> list[AuditTokenRow]:
    import torch

    prefix_ids = encode_prefix(tokenizer, sample["prefix"])
    word_ids = encode_target_word(tokenizer, sample["word"])
    current_ids = list(prefix_ids)
    rows: list[AuditTokenRow] = []

    model_device = device or str(next(model.parameters()).device)

    for token_index, token_id in enumerate(word_ids):
        input_ids = torch.tensor([current_ids], dtype=torch.long, device=model_device)
        with torch.inference_mode():
            outputs = model(input_ids=input_ids)
            logits = outputs.logits[0, -1, :]
        rank, probability, top_probability, ratio, cumulative_probability = (
            rank_probability_from_logits(logits, token_id)
        )
        rows.append(
            AuditTokenRow(
                sample_id=sample["id"],
                model=model_name,
                word=sample["word"],
                word_rank=int(sample["rank"]),
                source_path=sample["source_path"],
                word_token_index=token_index,
                token_id=int(token_id),
                token_text=tokenizer.decode([token_id]),
                rank=rank,
                probability=probability,
                top_probability=top_probability,
                probability_ratio_to_top=ratio,
                cumulative_probability=cumulative_probability,
                context_token_count=int(sample["context_token_count"]),
                prefix_char_count=len(sample["prefix"]),
                word_token_count=len(word_ids),
            )
        )
        current_ids.append(token_id)

    return rows


def audit_samples(
    model: Any,
    tokenizer: Any,
    samples: Iterable[dict[str, Any]],
    model_name: str,
    device: str | None = None,
) -> Iterator[AuditTokenRow]:
    for sample in samples:
        yield from audit_sample(model, tokenizer, sample, model_name=model_name, device=device)


def write_audit_jsonl(rows: Iterable[AuditTokenRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")


def load_hf_model_and_tokenizer(
    model_name: str,
    device: str,
    dtype: str = "auto",
    trust_remote_code: bool = False,
) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype: Any = dtype
    if dtype == "float16":
        torch_dtype = torch.float16
    elif dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    elif dtype == "float32":
        torch_dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
    )
    model.to(device)
    model.eval()
    return model, tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run WCS forced-path audit for one model.")
    parser.add_argument("--samples", type=Path, default=Path("data/processed/samples.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True, help="Hugging Face model id or local model path.")
    parser.add_argument("--device", default="cuda", help="Example: cuda, cuda:0, cpu")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--limit", type=int, default=None, help="Optional sample limit for smoke tests.")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model, tokenizer = load_hf_model_and_tokenizer(
        args.model,
        device=args.device,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
    )
    samples = load_samples(args.samples, limit=args.limit)
    rows = audit_samples(
        model=model,
        tokenizer=tokenizer,
        samples=samples,
        model_name=args.model,
        device=args.device,
    )
    write_audit_jsonl(rows, args.output)
    print(f"Wrote audit rows to {args.output}")


if __name__ == "__main__":
    main()
