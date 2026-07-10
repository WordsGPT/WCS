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


AUDIT_SCHEMA_VERSION = 2
DEFAULT_TOP_K = 5
DEFAULT_RANK_NEIGHBORS = 5


@dataclass(frozen=True)
class RankedToken:
    rank: int
    token_id: int
    probability: float


@dataclass(frozen=True)
class TokenPrediction:
    rank: int
    token_id: int
    token_text: str
    probability: float


@dataclass(frozen=True)
class RankProbabilityResult:
    rank: int
    probability: float
    top_probability: float
    probability_ratio_to_top: float
    cumulative_probability: float
    top_tokens: list[RankedToken]
    neighbors_above: list[RankedToken]
    neighbors_below: list[RankedToken]


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
    temperature: float
    context_token_count: int
    prefix_char_count: int
    word_token_count: int
    top_5_tokens: list[str] | None = None
    top_5_probs: list[float] | None = None
    rank_neighbors_above: list[TokenPrediction] | None = None
    rank_neighbors_below: list[TokenPrediction] | None = None
    rank_neighbor_count: int = 0
    audit_schema_version: int = AUDIT_SCHEMA_VERSION


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


def model_forward_no_cache(model: Any, input_ids: Any) -> Any:
    attempts = (
        {"use_cache": False, "logits_to_keep": 1},
        {"use_cache": False},
        {},
    )
    for index, kwargs in enumerate(attempts):
        try:
            return model(input_ids=input_ids, **kwargs)
        except TypeError as error:
            if index == len(attempts) - 1:
                raise
            if not any(name in str(error) for name in kwargs):
                raise
    raise AssertionError("unreachable")


def rank_probability_from_logits(
    logits: Any,
    token_id: int,
    temperature: float = 1.0,
    top_k: int = DEFAULT_TOP_K,
    neighbor_count: int = DEFAULT_RANK_NEIGHBORS,
) -> RankProbabilityResult:
    """Return rank/probability diagnostics for one token id.

    The neighbors are the tokens immediately above and below the target in the
    complete next-token distribution. They are different from the global top-k
    whenever the target is outside the top-k.
    """

    import torch

    if temperature <= 0:
        raise ValueError("temperature must be greater than 0")
    if top_k < 0:
        raise ValueError("top_k must be non-negative")
    if neighbor_count < 0:
        raise ValueError("neighbor_count must be non-negative")

    probs = torch.softmax(logits.float() / temperature, dim=-1)
    target_prob = probs[token_id]
    top_prob = torch.max(probs)
    rank = int(torch.count_nonzero(probs > target_prob).item() + 1)
    sorted_probs, sorted_ids = torch.sort(probs, descending=True)
    cumulative_probability = float(torch.sum(sorted_probs[:rank]).item())
    ratio = float((target_prob / top_prob).item()) if float(top_prob.item()) > 0 else 0.0

    target_positions = torch.nonzero(sorted_ids == token_id, as_tuple=False)
    if target_positions.numel() != 1:
        raise ValueError(f"Could not locate token id {token_id} in sorted vocabulary")
    target_position = int(target_positions[0].item())

    def ranked_slice(start: int, end: int) -> list[RankedToken]:
        return [
            RankedToken(
                rank=index + 1,
                token_id=int(sorted_ids[index].item()),
                probability=float(sorted_probs[index].item()),
            )
            for index in range(start, end)
        ]

    top_end = min(top_k, probs.size(-1))
    above_start = max(0, target_position - neighbor_count)
    below_end = min(probs.size(-1), target_position + 1 + neighbor_count)

    return RankProbabilityResult(
        rank=rank,
        probability=float(target_prob.item()),
        top_probability=float(top_prob.item()),
        probability_ratio_to_top=ratio,
        cumulative_probability=cumulative_probability,
        top_tokens=ranked_slice(0, top_end),
        neighbors_above=ranked_slice(above_start, target_position),
        neighbors_below=ranked_slice(target_position + 1, below_end),
    )


def decode_ranked_tokens(tokenizer: Any, tokens: list[RankedToken]) -> list[TokenPrediction]:
    return [
        TokenPrediction(
            rank=token.rank,
            token_id=token.token_id,
            token_text=tokenizer.decode([token.token_id]),
            probability=token.probability,
        )
        for token in tokens
    ]


def audit_sample(
    model: Any,
    tokenizer: Any,
    sample: dict[str, Any],
    model_name: str,
    device: str | None = None,
    temperature: float = 1.0,
    top_k: int = DEFAULT_TOP_K,
    rank_neighbors: int = DEFAULT_RANK_NEIGHBORS,
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
            outputs = model_forward_no_cache(model, input_ids)
            logits = outputs.logits[0, -1, :]
        result = rank_probability_from_logits(
            logits,
            token_id,
            temperature=temperature,
            top_k=top_k,
            neighbor_count=rank_neighbors,
        )
        top_predictions = decode_ranked_tokens(tokenizer, result.top_tokens)
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
                rank=result.rank,
                probability=result.probability,
                top_probability=result.top_probability,
                probability_ratio_to_top=result.probability_ratio_to_top,
                cumulative_probability=result.cumulative_probability,
                temperature=float(temperature),
                context_token_count=int(sample["context_token_count"]),
                prefix_char_count=len(sample["prefix"]),
                word_token_count=len(word_ids),
                top_5_tokens=[prediction.token_text for prediction in top_predictions],
                top_5_probs=[prediction.probability for prediction in top_predictions],
                rank_neighbors_above=decode_ranked_tokens(tokenizer, result.neighbors_above),
                rank_neighbors_below=decode_ranked_tokens(tokenizer, result.neighbors_below),
                rank_neighbor_count=rank_neighbors,
            )
        )
        current_ids.append(token_id)

    return rows


def audit_sample_temperatures(
    model: Any,
    tokenizer: Any,
    sample: dict[str, Any],
    model_name: str,
    temperatures: Iterable[float],
    device: str | None = None,
    top_k: int = DEFAULT_TOP_K,
    rank_neighbors: int = DEFAULT_RANK_NEIGHBORS,
) -> dict[float, list[AuditTokenRow]]:
    import torch

    temperature_values = [float(temperature) for temperature in temperatures]
    prefix_ids = encode_prefix(tokenizer, sample["prefix"])
    word_ids = encode_target_word(tokenizer, sample["word"])
    current_ids = list(prefix_ids)
    rows_by_temperature: dict[float, list[AuditTokenRow]] = {
        temperature: [] for temperature in temperature_values
    }

    model_device = device or str(next(model.parameters()).device)

    for token_index, token_id in enumerate(word_ids):
        input_ids = torch.tensor([current_ids], dtype=torch.long, device=model_device)
        with torch.inference_mode():
            outputs = model_forward_no_cache(model, input_ids)
            logits = outputs.logits[0, -1, :]
        for temperature in temperature_values:
            result = rank_probability_from_logits(
                logits,
                token_id,
                temperature=temperature,
                top_k=top_k,
                neighbor_count=rank_neighbors,
            )
            top_predictions = decode_ranked_tokens(tokenizer, result.top_tokens)
            rows_by_temperature[temperature].append(
                AuditTokenRow(
                    sample_id=sample["id"],
                    model=model_name,
                    word=sample["word"],
                    word_rank=int(sample["rank"]),
                    source_path=sample["source_path"],
                    word_token_index=token_index,
                    token_id=int(token_id),
                    token_text=tokenizer.decode([token_id]),
                    rank=result.rank,
                    probability=result.probability,
                    top_probability=result.top_probability,
                    probability_ratio_to_top=result.probability_ratio_to_top,
                    cumulative_probability=result.cumulative_probability,
                    temperature=float(temperature),
                    context_token_count=int(sample["context_token_count"]),
                    prefix_char_count=len(sample["prefix"]),
                    word_token_count=len(word_ids),
                    top_5_tokens=[prediction.token_text for prediction in top_predictions],
                    top_5_probs=[prediction.probability for prediction in top_predictions],
                    rank_neighbors_above=decode_ranked_tokens(
                        tokenizer, result.neighbors_above
                    ),
                    rank_neighbors_below=decode_ranked_tokens(
                        tokenizer, result.neighbors_below
                    ),
                    rank_neighbor_count=rank_neighbors,
                )
            )
        current_ids.append(token_id)

    return rows_by_temperature


def audit_samples(
    model: Any,
    tokenizer: Any,
    samples: Iterable[dict[str, Any]],
    model_name: str,
    device: str | None = None,
    temperature: float = 1.0,
    top_k: int = DEFAULT_TOP_K,
    rank_neighbors: int = DEFAULT_RANK_NEIGHBORS,
) -> Iterator[AuditTokenRow]:
    for sample in samples:
        yield from audit_sample(
            model,
            tokenizer,
            sample,
            model_name=model_name,
            device=device,
            temperature=temperature,
            top_k=top_k,
            rank_neighbors=rank_neighbors,
        )


def audit_samples_temperatures(
    model: Any,
    tokenizer: Any,
    samples: Iterable[dict[str, Any]],
    model_name: str,
    temperatures: Iterable[float],
    device: str | None = None,
    top_k: int = DEFAULT_TOP_K,
    rank_neighbors: int = DEFAULT_RANK_NEIGHBORS,
) -> Iterator[tuple[float, AuditTokenRow]]:
    temperature_values = [float(temperature) for temperature in temperatures]
    for sample in samples:
        rows_by_temperature = audit_sample_temperatures(
            model=model,
            tokenizer=tokenizer,
            sample=sample,
            model_name=model_name,
            temperatures=temperature_values,
            device=device,
            top_k=top_k,
            rank_neighbors=rank_neighbors,
        )
        for temperature in temperature_values:
            for row in rows_by_temperature[temperature]:
                yield temperature, row


def write_audit_jsonl(
    rows: Iterable[AuditTokenRow], output_path: Path, *, append: bool = False
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a" if append else "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")
            handle.flush()


def write_audit_jsonl_by_temperature(
    rows: Iterable[tuple[float, AuditTokenRow]],
    output_paths: dict[float, Path],
    *,
    append: bool = False,
) -> None:
    handles = {}
    try:
        for temperature, output_path in output_paths.items():
            output_path.parent.mkdir(parents=True, exist_ok=True)
            handles[temperature] = output_path.open(
                "a" if append else "w", encoding="utf-8"
            )
        for temperature, row in rows:
            handles[temperature].write(json.dumps(asdict(row), ensure_ascii=False) + "\n")
            handles[temperature].flush()
    finally:
        for handle in handles.values():
            handle.close()


def parse_temperature_list(raw: str) -> list[float]:
    values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("At least one temperature is required")
    if any(value <= 0 for value in values):
        raise ValueError("All temperatures must be greater than 0")
    return values


def temperature_slug(temperature: float) -> str:
    return f"t{temperature:g}".replace(".", "p")


def completed_sample_ids(path: Path) -> set[str]:
    """Return sample ids whose token rows form one complete forced path."""
    if not path.exists():
        return set()
    groups: dict[str, list[dict[str, Any]]] = {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                    groups.setdefault(str(row["sample_id"]), []).append(row)
                except (json.JSONDecodeError, KeyError):
                    continue
    except OSError:
        return set()
    complete: set[str] = set()
    for sample_id, rows in groups.items():
        expected = int(rows[0].get("word_token_count", 0))
        indices = {int(row["word_token_index"]) for row in rows}
        if expected > 0 and indices == set(range(expected)) and len(rows) == expected:
            complete.add(sample_id)
    return complete


def retain_complete_samples(path: Path, sample_ids: set[str]) -> None:
    """Atomically discard malformed, partial, duplicate, or non-common groups."""
    if not path.exists():
        return
    groups: dict[str, list[dict[str, Any]]] = {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                    groups.setdefault(str(row["sample_id"]), []).append(row)
                except (json.JSONDecodeError, KeyError):
                    continue
    except OSError:
        groups = {}
    temporary = path.with_suffix(path.suffix + ".resume-tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for sample_id, rows in groups.items():
            if sample_id not in sample_ids:
                continue
            expected = int(rows[0].get("word_token_count", 0))
            by_index = {int(row["word_token_index"]): row for row in rows}
            if expected <= 0 or set(by_index) != set(range(expected)):
                continue
            for index in range(expected):
                handle.write(json.dumps(by_index[index], ensure_ascii=False) + "\n")
    temporary.replace(path)


def load_hf_model_and_tokenizer(
    model_name: str,
    device: str,
    dtype: str = "auto",
    trust_remote_code: bool = False,
    device_map: str | None = None,
    max_memory: dict[Any, str] | None = None,
    offload_folder: str | None = None,
) -> tuple[Any, Any]:
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForMultimodalLM,
        AutoProcessor,
        AutoTokenizer,
    )

    if trust_remote_code:
        patch_transformers_remote_code_compatibility()

    torch_dtype: Any = dtype
    if dtype == "float16":
        torch_dtype = torch.float16
    elif dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    elif dtype == "float32":
        torch_dtype = torch.float32

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        )
    except (OSError, TypeError, ValueError):
        processor = AutoProcessor.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        )
        tokenizer = getattr(processor, "tokenizer", processor)
        if not hasattr(tokenizer, "encode") or not hasattr(tokenizer, "decode"):
            raise TypeError(
                f"{model_name} processor does not expose a text tokenizer"
            )
    model_kwargs: dict[str, Any] = {
        "dtype": torch_dtype,
        "trust_remote_code": trust_remote_code,
    }
    if device_map:
        model_kwargs["device_map"] = device_map
        model_kwargs["low_cpu_mem_usage"] = True
    if max_memory:
        model_kwargs["max_memory"] = max_memory
    if offload_folder:
        model_kwargs["offload_folder"] = offload_folder

    def from_pretrained(model_class: Any, kwargs: dict[str, Any]) -> Any:
        try:
            return model_class.from_pretrained(model_name, **kwargs)
        except TypeError as error:
            if "dtype" not in str(error) or "dtype" not in kwargs:
                raise
            legacy_kwargs = dict(kwargs)
            legacy_kwargs["torch_dtype"] = legacy_kwargs.pop("dtype")
            return model_class.from_pretrained(model_name, **legacy_kwargs)

    try:
        model = from_pretrained(AutoModelForCausalLM, model_kwargs)
    except ValueError as error:
        if "Unrecognized configuration class" not in str(error):
            raise
        model = from_pretrained(AutoModelForMultimodalLM, model_kwargs)
    if not device_map:
        model.to(device)
    model.eval()
    return model, tokenizer


def parse_max_memory(raw: str | None) -> dict[Any, str] | None:
    if not raw:
        return None
    values: dict[Any, str] = {}
    for part in raw.split(","):
        if not part.strip():
            continue
        key, separator, value = part.partition("=")
        if not separator:
            raise ValueError("--max-memory entries must look like 0=44GiB,1=44GiB,cpu=160GiB")
        clean_key: Any = key.strip()
        if clean_key.startswith("cuda:"):
            clean_key = clean_key.removeprefix("cuda:")
        if isinstance(clean_key, str) and clean_key.isdigit():
            clean_key = int(clean_key)
        values[clean_key] = value.strip()
    return values


def patch_transformers_remote_code_compatibility() -> None:
    """Restore small deprecated helpers referenced by older Hub remote code."""

    import transformers.pytorch_utils as pytorch_utils
    import transformers.utils.import_utils as import_utils

    if not hasattr(import_utils, "is_torch_fx_available"):
        import_utils.is_torch_fx_available = lambda: True

    if not hasattr(pytorch_utils, "is_torch_greater_or_equal_than_1_13"):
        pytorch_utils.is_torch_greater_or_equal_than_1_13 = True

    try:
        from transformers.cache_utils import DynamicCache
    except Exception:
        return

    if not hasattr(DynamicCache, "from_legacy_cache"):
        def from_legacy_cache(cls: type[Any], past_key_values: Any = None, *args: Any, **kwargs: Any) -> Any:
            if past_key_values is None or isinstance(past_key_values, cls):
                return past_key_values if isinstance(past_key_values, cls) else cls()
            cache = cls()
            update = getattr(cache, "update", None)
            if update is None:
                return past_key_values
            for layer_idx, layer_past in enumerate(past_key_values):
                if not layer_past:
                    continue
                update(layer_past[0], layer_past[1], layer_idx)
            return cache

        DynamicCache.from_legacy_cache = classmethod(from_legacy_cache)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run WCS forced-path audit for one model.")
    parser.add_argument("--samples", type=Path, default=Path("data/processed/samples.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--output-template",
        default=None,
        help="Template for multi-temperature output; supports {temperature} and {temperature_slug}.",
    )
    parser.add_argument("--model", required=True, help="Hugging Face model id or local model path.")
    parser.add_argument("--device", default="cuda", help="Example: cuda, cuda:0, cpu")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--device-map", default=None, help="Optional Transformers device_map, e.g. auto.")
    parser.add_argument(
        "--max-memory",
        default=None,
        help="Optional comma list for device_map, e.g. 0=44GiB,1=44GiB,cpu=160GiB.",
    )
    parser.add_argument("--offload-folder", default=None, help="Optional folder for unquantized CPU/disk offload.")
    parser.add_argument("--limit", type=int, default=None, help="Optional sample limit for smoke tests.")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--temperatures", default=None, help="Comma-separated temperatures, e.g. 1.0,0.7,1.5")
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of global highest-probability tokens to save per target token.",
    )
    parser.add_argument(
        "--rank-neighbors",
        type=int,
        default=DEFAULT_RANK_NEIGHBORS,
        help="Number of tokens immediately above and below the target rank to save.",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append after complete samples already present in the output file(s).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.top_k < 0:
        raise ValueError("--top-k must be non-negative")
    if args.rank_neighbors < 0:
        raise ValueError("--rank-neighbors must be non-negative")
    model, tokenizer = load_hf_model_and_tokenizer(
        args.model,
        device=args.device,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        device_map=args.device_map,
        max_memory=parse_max_memory(args.max_memory),
        offload_folder=args.offload_folder,
    )
    samples = load_samples(args.samples, limit=args.limit)
    if args.temperatures:
        if not args.output_template:
            raise ValueError("--output-template is required when --temperatures is used")
        temperatures = parse_temperature_list(args.temperatures)
        output_paths = {
            temperature: Path(
                args.output_template.format(
                    temperature=f"{temperature:g}",
                    temperature_slug=temperature_slug(temperature),
                )
            )
            for temperature in temperatures
        }
        if args.resume:
            completed_sets = [completed_sample_ids(path) for path in output_paths.values()]
            completed = set.intersection(*completed_sets) if completed_sets else set()
            for path in output_paths.values():
                retain_complete_samples(path, completed)
            samples = [sample for sample in samples if str(sample["id"]) not in completed]
            print(f"Resuming after {len(completed)} complete samples; {len(samples)} remain")
        rows = audit_samples_temperatures(
            model=model,
            tokenizer=tokenizer,
            samples=samples,
            model_name=args.model,
            temperatures=temperatures,
            device=args.device,
            top_k=args.top_k,
            rank_neighbors=args.rank_neighbors,
        )
        write_audit_jsonl_by_temperature(rows, output_paths, append=args.resume)
        for temperature, output_path in output_paths.items():
            print(f"Wrote T={temperature:g} audit rows to {output_path}")
    else:
        if args.resume:
            completed = completed_sample_ids(args.output)
            retain_complete_samples(args.output, completed)
            samples = [sample for sample in samples if str(sample["id"]) not in completed]
            print(f"Resuming after {len(completed)} complete samples; {len(samples)} remain")
        rows = audit_samples(
            model=model,
            tokenizer=tokenizer,
            samples=samples,
            model_name=args.model,
            device=args.device,
            temperature=args.temperature,
            top_k=args.top_k,
            rank_neighbors=args.rank_neighbors,
        )
        write_audit_jsonl(rows, args.output, append=args.resume)
        print(f"Wrote audit rows to {args.output}")


if __name__ == "__main__":
    main()
