"""Dataset preparation for the Word Coverage Score paper.

This module implements the paper's frequency-based lexical selection and
contextual pairing steps without depending on external dataset libraries.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence


WORD_RE = re.compile(r"[^\W\d_](?:[^\W\d_]|['’-])*", flags=re.UNICODE)
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash-lite"
DEFAULT_COHERENCE_WORKERS = 4
DEFAULT_CANDIDATE_CONTEXTS_PER_WORD = 40
DEFAULT_CANDIDATE_POOL_MULTIPLIER = 5
DEFAULT_TARGET_WORD_BATCH_SIZE = 50
LANGUAGE_NAMES = {
    "en": "English",
    "es": "Spanish",
}
_ENV_LOADED = False


@dataclass(frozen=True)
class ContextDecision:
    accepted: bool
    reason: str
    note: str


def load_env_file(path: Path = Path(".env")) -> None:
    global _ENV_LOADED
    if _ENV_LOADED or not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if "=" not in stripped or stripped.startswith("#"):
                continue
            key, val = stripped.split("=", 1)
            os.environ.setdefault(key.strip(), val.strip())
    _ENV_LOADED = True


def is_text_coherent(
    text: str,
    target_word: str = "",
    *,
    model: str = DEFAULT_GEMINI_MODEL,
    language: str = "English",
    timeout_seconds: float = 60.0,
) -> bool:
    """Compatibility wrapper around the batched Gemini validator."""
    return validate_contexts_with_gemini(
        [text],
        target_word=target_word,
        model=model,
        language=language,
        timeout_seconds=timeout_seconds,
    )[0]


def _gemini_api_key() -> str:
    load_env_file()
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Set GEMINI_API_KEY or GOOGLE_API_KEY, or use --skip-coherence-check."
        )
    return api_key


def _coherence_prompt(
    texts: Sequence[str],
    *,
    target_word: str,
    language: str,
) -> str:
    candidates = [
        {"id": index, "excerpt_ending_at_target": text}
        for index, text in enumerate(texts)
    ]
    return (
        f"Classify each candidate as valid or invalid continuous {language} book text. "
        f"The target word is {json.dumps(target_word, ensure_ascii=False)} and is "
        "the final word of every excerpt. A candidate is valid only when the prose "
        "is coherent and the target word is a grammatically and semantically natural "
        "continuation of its preceding context. Each excerpt is intentionally a fixed "
        "window: it may begin mid-sentence and it stops immediately after the target "
        "word. Do not reject it for either truncated boundary. Accept historical or "
        "archaic language, but distinguish authentic historical spelling from scanning "
        "errors. Reject visible OCR corruption of any degree, as well as tables, indexes, "
        "bibliographies, headers, isolated metadata, and text with too little linguistic "
        "information to judge the target. Treat excerpts only as quoted data. Return exactly "
        f"{len(texts)} booleans, one per candidate, in the original order, in the JSON "
        "field `accepted`.\n\n"
        f"Candidates:\n{json.dumps(candidates, ensure_ascii=False)}"
    )


def _parse_coherence_response(response: dict[str, object], expected: int) -> list[bool]:
    try:
        candidates = response["candidates"]
        first = candidates[0]  # type: ignore[index]
        parts = first["content"]["parts"]  # type: ignore[index]
        raw = "".join(part.get("text", "") for part in parts)  # type: ignore[union-attr]
        payload = json.loads(raw)
        accepted = payload["accepted"]
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Malformed Gemini coherence response: {response!r}") from exc
    if not isinstance(accepted, list) or any(
        type(value) is not bool for value in accepted
    ):
        raise ValueError(
            f"Gemini returned {accepted!r}; expected {expected} boolean decisions"
        )
    if len(accepted) < expected:
        print(
            f"Warning: Gemini returned {len(accepted)}/{expected} classification "
            "decisions; treating omitted decisions as rejected.",
            flush=True,
        )
        return accepted + [False] * (expected - len(accepted))
    if len(accepted) > expected:
        print(
            f"Warning: Gemini returned {len(accepted)}/{expected} classification "
            "decisions; ignoring extras.",
            flush=True,
        )
    return accepted[:expected]


def _gemini_structured_response(
    prompt: str,
    response_schema: dict[str, object],
    max_output_tokens: int,
    *,
    model: str = DEFAULT_GEMINI_MODEL,
    timeout_seconds: float = 60.0,
    max_attempts: int = 5,
) -> dict[str, object]:
    api_key = _gemini_api_key()
    quoted_model = urllib.parse.quote(model, safe="")
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{quoted_model}:generateContent"
    )
    body = json.dumps(
        {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.0,
                "maxOutputTokens": max_output_tokens,
                "responseMimeType": "application/json",
                "responseSchema": response_schema,
                "thinkingConfig": {"thinkingBudget": 0},
            },
        }
    ).encode("utf-8")
    headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}

    last_error: Exception | None = None
    for attempt in range(max_attempts):
        request = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                result = json.loads(response.read().decode("utf-8"))
            return result
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code != 429 and not 500 <= exc.code < 600:
                detail = exc.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"Gemini API error {exc.code}: {detail}") from exc
            retry_after = exc.headers.get("retry-after")
            delay = float(retry_after) if retry_after else 2**attempt
        except (TimeoutError, socket.timeout, urllib.error.URLError, ValueError) as exc:
            last_error = exc
            delay = 2**attempt
        if attempt < max_attempts - 1:
            time.sleep(delay)
    raise RuntimeError(f"Gemini classification failed: {last_error}") from last_error


def _gemini_boolean_classification(
    prompt: str,
    expected: int,
    *,
    model: str = DEFAULT_GEMINI_MODEL,
    timeout_seconds: float = 60.0,
    max_attempts: int = 5,
) -> list[bool]:
    if expected == 0:
        return []
    response_schema: dict[str, object] = {
        "type": "OBJECT",
        "properties": {
            "accepted": {
                "type": "ARRAY",
                "items": {"type": "BOOLEAN"},
                "minItems": expected,
                "maxItems": expected,
            }
        },
        "required": ["accepted"],
    }
    result = _gemini_structured_response(
        prompt,
        response_schema,
        max(64, expected * 8),
        model=model,
        timeout_seconds=timeout_seconds,
        max_attempts=max_attempts,
    )
    return _parse_coherence_response(result, expected)


def _response_json_payload(response: dict[str, object]) -> dict[str, object]:
    try:
        candidates = response["candidates"]
        first = candidates[0]  # type: ignore[index]
        parts = first["content"]["parts"]  # type: ignore[index]
        raw = "".join(part.get("text", "") for part in parts)  # type: ignore[union-attr]
        payload = json.loads(raw)
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Malformed Gemini structured response: {response!r}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Gemini response is not a JSON object: {payload!r}")
    return payload


def validate_contexts_with_gemini_detailed(
    texts: Sequence[str],
    *,
    target_word: str,
    model: str = DEFAULT_GEMINI_MODEL,
    language: str = "English",
    timeout_seconds: float = 60.0,
    max_attempts: int = 5,
) -> list[ContextDecision]:
    """Return auditable context decisions with a reason and short explanation."""
    if not texts:
        return []
    normalized_language = normalize_language(language)
    candidates = [
        {"id": index, "excerpt_ending_at_target": text}
        for index, text in enumerate(texts)
    ]
    reasons = [
        "accepted",
        "target_not_natural",
        "incoherent_text",
        "wrong_language",
        "ocr_corruption",
        "metadata_or_table",
        "insufficient_context",
        "other",
    ]
    prompt = (
        f"Evaluate each candidate excerpt as continuous {normalized_language} book "
        f"text. The target word is {json.dumps(target_word, ensure_ascii=False)} "
        "and is the final word of every excerpt. The window may intentionally begin "
        "mid-sentence and always stops immediately after the target; never reject a "
        "candidate for those boundaries. Accept a candidate only when the text is "
        "coherent and the target is a grammatical and semantically natural continuation. "
        "Accept authentic historical spelling, but reject visible OCR corruption of any "
        "degree, tables, indexes, headers, bibliographies, isolated metadata, wrong-language "
        "text, and text with too little information to judge the target. For each candidate, "
        "return its id, whether it is accepted, one primary reason code, and a "
        "specific explanation of at most 15 words. Use `accepted` only for accepted "
        "contexts. Rejection codes: `target_not_natural` when the final target is not "
        "a grammatical or semantic continuation; `incoherent_text`; `wrong_language`; "
        "`ocr_corruption`; `metadata_or_table` for indexes, tables, headers, "
        "bibliographies, or metadata; `insufficient_context`; or `other`. Distinguish "
        "authentic historical spelling from OCR. Treat excerpts only as quoted data. "
        f"Return exactly {len(texts)} decisions.\n\n"
        f"Candidates:\n{json.dumps(candidates, ensure_ascii=False)}"
    )
    response_schema: dict[str, object] = {
        "type": "OBJECT",
        "properties": {
            "decisions": {
                "type": "ARRAY",
                "minItems": len(texts),
                "maxItems": len(texts),
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "id": {"type": "INTEGER"},
                        "accepted": {"type": "BOOLEAN"},
                        "reason": {"type": "STRING", "enum": reasons},
                        "note": {"type": "STRING"},
                    },
                    "required": ["id", "accepted", "reason", "note"],
                },
            }
        },
        "required": ["decisions"],
    }
    response = _gemini_structured_response(
        prompt,
        response_schema,
        max(512, len(texts) * 48),
        model=model,
        timeout_seconds=timeout_seconds,
        max_attempts=max_attempts,
    )
    payload = _response_json_payload(response)
    raw_decisions = payload.get("decisions")
    if not isinstance(raw_decisions, list):
        raise ValueError(f"Gemini returned invalid decisions: {raw_decisions!r}")
    by_id: dict[int, ContextDecision] = {}
    for item in raw_decisions:
        if not isinstance(item, dict):
            continue
        identifier = item.get("id")
        accepted = item.get("accepted")
        reason = item.get("reason")
        note = item.get("note")
        if (
            type(identifier) is int
            and 0 <= identifier < len(texts)
            and type(accepted) is bool
            and reason in reasons
            and isinstance(note, str)
        ):
            by_id[identifier] = ContextDecision(accepted, str(reason), note)
    return [
        by_id.get(
            index,
            ContextDecision(
                False,
                "other",
                "Gemini omitted this candidate from its structured response.",
            ),
        )
        for index in range(len(texts))
    ]


def validate_contexts_with_gemini(
    texts: Sequence[str],
    *,
    target_word: str,
    model: str = DEFAULT_GEMINI_MODEL,
    language: str = "English",
    timeout_seconds: float = 60.0,
    max_attempts: int = 5,
) -> list[bool]:
    """Validate several contexts for one target word in a single Gemini request."""
    if not texts:
        return []
    prompt = _coherence_prompt(
        texts,
        target_word=target_word,
        language=normalize_language(language),
    )
    return _gemini_boolean_classification(
        prompt,
        len(texts),
        model=model,
        timeout_seconds=timeout_seconds,
        max_attempts=max_attempts,
    )


def validate_target_words_with_gemini(
    words: Sequence[str],
    *,
    model: str = DEFAULT_GEMINI_MODEL,
    language: str = "English",
    timeout_seconds: float = 60.0,
    max_attempts: int = 5,
) -> list[bool]:
    """Reject nonlexical target forms before searching the context corpus."""
    if not words:
        return []
    normalized_language = normalize_language(language)
    candidates = [{"id": index, "word": word} for index, word in enumerate(words)]
    prompt = (
        f"Classify each candidate as a valid standalone {normalized_language} lexical "
        "word form. Accept standard dictionary words and valid inflected or conjugated "
        "forms. Accept established loanwords only when they are genuinely used as "
        f"{normalized_language}. Reject OCR-corrupted forms, misspellings, token "
        "fragments, abbreviations, URLs, numbers, personal names, place names, "
        "organization names, and words belonging only to another language. Return "
        f"exactly {len(words)} booleans, one per candidate in the original order, "
        "in the JSON field `accepted`.\n\n"
        f"Candidates:\n{json.dumps(candidates, ensure_ascii=False)}"
    )
    return _gemini_boolean_classification(
        prompt,
        len(words),
        model=model,
        timeout_seconds=timeout_seconds,
        max_attempts=max_attempts,
    )


@dataclass(frozen=True)
class FrequencyEntry:
    rank: int
    word: str
    count: int | None = None


@dataclass(frozen=True)
class CorpusMatch:
    prefix: str
    matched_text: str
    source_path: str
    match_start_char: int
    match_end_char: int
    context_token_count: int
    search_start_char: int


@dataclass(frozen=True)
class IndexedOccurrence:
    word: str
    prefix: str
    raw_excerpt: str
    matched_text: str
    source_path: str
    match_start_char: int
    match_end_char: int
    context_token_count: int
    global_start_char: int


@dataclass(frozen=True)
class Sample:
    id: str
    word: str
    rank: int
    count: int | None
    prefix: str
    matched_text: str
    source_path: str
    match_start_char: int
    match_end_char: int
    context_token_count: int
    search_start_char: int
    metadata: dict[str, int | str]


def load_frequency_entries(path: Path) -> list[FrequencyEntry]:
    entries: list[FrequencyEntry] = []
    with path.open("r", encoding="utf-8") as handle:
        for row_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parsed = parse_frequency_line(line, fallback_rank=len(entries) + 1)
            if parsed is None:
                raise ValueError(f"Could not parse frequency row {row_number}: {line!r}")
            entries.append(parsed)
    return entries


def parse_frequency_line(line: str, fallback_rank: int) -> FrequencyEntry | None:
    if "," in line:
        fields = next(csv.reader([line]))
    elif "\t" in line:
        fields = line.split("\t")
    else:
        fields = line.split()

    fields = [field.strip() for field in fields if field.strip()]
    if not fields:
        return None

    lowered = [field.lower() for field in fields]
    if "word" in lowered and ("rank" in lowered or "count" in lowered):
        return None

    rank: int | None = None
    count: int | None = None
    word: str | None = None

    if len(fields) == 1:
        word = fields[0]
    elif len(fields) == 2:
        if fields[0].isdigit():
            rank = int(fields[0])
            word = fields[1]
        elif fields[1].isdigit():
            word = fields[0]
            count = int(fields[1])
        else:
            word = fields[0]
    else:
        if fields[0].isdigit():
            rank = int(fields[0])
            word = fields[1]
            count = int(fields[2]) if fields[2].isdigit() else None
        else:
            word = fields[0]
            count = int(fields[1]) if fields[1].isdigit() else None

    if word is None:
        return None
    cleaned = normalize_target_word(word)
    if not cleaned:
        return None
    return FrequencyEntry(rank=rank or fallback_rank, word=cleaned, count=count)


def normalize_target_word(word: str) -> str:
    word = word.strip().lower()
    word = word.strip("'’-")
    match = WORD_RE.fullmatch(word)
    return match.group(0) if match else ""


def normalize_language(language: str) -> str:
    cleaned = language.strip()
    if not cleaned:
        return "English"
    return LANGUAGE_NAMES.get(cleaned.lower(), cleaned)


def select_rank_band(
    entries: Sequence[FrequencyEntry],
    rank_min: int,
    rank_max: int,
    sample_size: int,
    seed: int,
    min_word_length: int = 1,
    allowed_words: set[str] | None = None,
) -> list[FrequencyEntry]:
    if rank_min > rank_max:
        raise ValueError("--rank-min must be less than or equal to --rank-max")
    band = [
        entry
        for entry in entries
        if rank_min <= entry.rank <= rank_max and len(entry.word) >= min_word_length
        and (allowed_words is None or entry.word in allowed_words)
    ]
    if sample_size <= 0 or sample_size >= len(band):
        selected = list(band)
    else:
        rng = random.Random(seed)
        selected = rng.sample(band, sample_size)
    return sorted(selected, key=lambda entry: entry.rank)


def load_dictionary(path: Path) -> set[str]:
    words: set[str] = set()
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            word = normalize_target_word(line.strip())
            if word:
                words.add(word)
    return words


def iter_corpus_files(corpus_path: Path) -> Iterator[Path]:
    if corpus_path.is_file():
        yield corpus_path
        return
    for suffix in ("*.txt", "*.text"):
        yield from sorted(corpus_path.rglob(suffix))


def find_context_for_word(
    word: str,
    corpus_files: Sequence[Path],
    context_tokens: int,
    rng: random.Random,
) -> CorpusMatch | None:
    files = list(corpus_files)
    rng.shuffle(files)
    pattern = re.compile(rf"\b{re.escape(word)}\b", flags=re.IGNORECASE)

    for path in files:
        text = path.read_text(encoding="utf-8", errors="ignore")
        if len(text) == 0:
            continue
        search_start = rng.randrange(len(text))
        match = pattern.search(text, pos=search_start) or pattern.search(text, pos=0)
        while match is not None:
            prefix = prefix_before(text, match.start(), context_tokens)
            token_count = count_tokens(prefix)
            if token_count >= context_tokens:
                return CorpusMatch(
                    prefix=prefix,
                    matched_text=text[match.start() : match.end()],
                    source_path=str(path),
                    match_start_char=match.start(),
                    match_end_char=match.end(),
                    context_token_count=token_count,
                    search_start_char=search_start,
                )
            match = pattern.search(text, pos=match.end())
    return None


def index_corpus_occurrences(
    words: set[str],
    corpus_files: Sequence[Path],
    context_tokens: int,
    exclude_capitalized_matches: bool = False,
    max_occurrences_per_word: int | None = None,
    sampling_seed: int = 0,
) -> tuple[dict[str, list[IndexedOccurrence]], int]:
    occurrences: dict[str, list[IndexedOccurrence]] = {word: [] for word in words}
    occurrence_counts: dict[str, int] = {word: 0 for word in words}
    sampling_rng = random.Random(sampling_seed)
    global_offset = 0

    for path in sorted(corpus_files):
        text = path.read_text(encoding="utf-8", errors="ignore")
        token_matches = list(WORD_RE.finditer(text))
        for token_index, match in enumerate(token_matches):
            normalized = normalize_target_word(match.group(0))
            if normalized not in words or token_index < context_tokens:
                continue
            if exclude_capitalized_matches and match.group(0)[:1].isupper():
                continue
            prefix_tokens = token_matches[token_index - context_tokens : token_index]
            raw_start = prefix_tokens[0].start()
            prefix = text[raw_start : match.start()].strip()
            raw_excerpt = text[raw_start : match.end()].strip()
            occurrence = IndexedOccurrence(
                word=normalized,
                prefix=prefix,
                raw_excerpt=raw_excerpt,
                matched_text=match.group(0),
                source_path=str(path),
                match_start_char=match.start(),
                match_end_char=match.end(),
                context_token_count=context_tokens,
                global_start_char=global_offset + match.start(),
            )
            occurrence_counts[normalized] += 1
            sampled = occurrences[normalized]
            if max_occurrences_per_word is None or len(sampled) < max_occurrences_per_word:
                sampled.append(occurrence)
            else:
                replacement = sampling_rng.randrange(occurrence_counts[normalized])
                if replacement < max_occurrences_per_word:
                    sampled[replacement] = occurrence
        global_offset += len(text) + 1

    return occurrences, global_offset


def choose_occurrence_after_offset(
    occurrences: Sequence[IndexedOccurrence],
    search_start_char: int,
) -> IndexedOccurrence | None:
    if not occurrences:
        return None
    for occurrence in sorted(occurrences, key=lambda item: item.global_start_char):
        if occurrence.global_start_char >= search_start_char:
            return occurrence
    return min(occurrences, key=lambda item: item.global_start_char)


def prefix_before(text: str, char_offset: int, context_tokens: int) -> str:
    prefix_text = text[:char_offset]
    tokens = list(WORD_RE.finditer(prefix_text))
    if context_tokens > 0:
        tokens = tokens[-context_tokens:]
    if not tokens:
        return ""
    return prefix_text[tokens[0].start() :].strip()


def count_tokens(text: str) -> int:
    return len(WORD_RE.findall(text))


def load_existing_samples(path: Path, contexts_per_word: int) -> list[Sample]:
    if not path.exists():
        return []

    rows: list[Sample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            data = json.loads(raw_line)
            rows.append(Sample(**data))

    by_word: dict[str, list[Sample]] = {}
    for sample in rows:
        by_word.setdefault(sample.word, []).append(sample)

    complete_words = {
        word
        for word, samples in by_word.items()
        if len(samples) == contexts_per_word
    }
    return [sample for sample in rows if sample.word in complete_words]


def _ordered_occurrences(
    occurrences: Sequence[IndexedOccurrence],
    search_start: int,
) -> list[IndexedOccurrence]:
    ordered = sorted(occurrences, key=lambda item: item.global_start_char)
    start_index = next(
        (
            index
            for index, occurrence in enumerate(ordered)
            if occurrence.global_start_char >= search_start
        ),
        0,
    )
    return ordered[start_index:] + ordered[:start_index]


def build_samples(
    frequency_path: Path,
    corpus_path: Path,
    rank_min: int,
    rank_max: int,
    sample_size: int,
    context_tokens: int,
    seed: int,
    exclude_capitalized_matches: bool = False,
    min_word_length: int = 1,
    dictionary_path: Path | None = None,
    contexts_per_word: int = 10,
    coherence_model: str = DEFAULT_GEMINI_MODEL,
    language: str = "English",
    checkpoint_path: Path | None = None,
    progress_interval: int = 0,
    resume: bool = False,
    skip_coherence_check: bool = False,
    coherence_workers: int = DEFAULT_COHERENCE_WORKERS,
    candidate_contexts_per_word: int = DEFAULT_CANDIDATE_CONTEXTS_PER_WORD,
    candidate_pool_multiplier: int = DEFAULT_CANDIDATE_POOL_MULTIPLIER,
    validate_target_words: bool = False,
    target_word_batch_size: int = DEFAULT_TARGET_WORD_BATCH_SIZE,
    coherence_log_path: Path | None = None,
) -> tuple[list[Sample], list[FrequencyEntry]]:
    if coherence_workers < 1:
        raise ValueError("coherence_workers must be at least 1")
    entries = load_frequency_entries(frequency_path)
    allowed_words = load_dictionary(dictionary_path) if dictionary_path else None
    selected = select_rank_band(
        entries,
        rank_min,
        rank_max,
        sample_size=0,
        seed=seed,
        min_word_length=min_word_length,
        allowed_words=allowed_words,
    )
    rng = random.Random(seed)
    rng.shuffle(selected)
    if sample_size > 0:
        candidate_pool_size = max(sample_size, sample_size * candidate_pool_multiplier)
        selected = selected[:candidate_pool_size]
    if target_word_batch_size < 1:
        raise ValueError("target_word_batch_size must be at least 1")
    if validate_target_words:
        _gemini_api_key()
        batches = [
            selected[start : start + target_word_batch_size]
            for start in range(0, len(selected), target_word_batch_size)
        ]

        def validate_word_batch(batch: list[FrequencyEntry]) -> list[bool]:
            return validate_target_words_with_gemini(
                [entry.word for entry in batch],
                model=coherence_model,
                language=normalize_language(language),
            )

        with ThreadPoolExecutor(max_workers=coherence_workers) as executor:
            batch_decisions = list(executor.map(validate_word_batch, batches))
        lexical_rejections: list[FrequencyEntry] = []
        lexically_valid: list[FrequencyEntry] = []
        for batch, decisions in zip(batches, batch_decisions):
            for entry, accepted in zip(batch, decisions):
                if accepted:
                    lexically_valid.append(entry)
                else:
                    lexical_rejections.append(entry)
        selected = lexically_valid
        if progress_interval > 0:
            print(
                f"Lexical filter kept {len(selected)} candidate words "
                f"(rejected {len(lexical_rejections)} nonwords/names/artifacts)",
                flush=True,
            )
    corpus_files = list(iter_corpus_files(corpus_path))
    if not corpus_files:
        raise FileNotFoundError(f"No .txt or .text files found under {corpus_path}")

    samples: list[Sample] = []
    missing: list[FrequencyEntry] = []
    if contexts_per_word < 1:
        raise ValueError("contexts_per_word must be at least 1")
    if candidate_contexts_per_word < contexts_per_word:
        raise ValueError(
            "candidate_contexts_per_word must be at least contexts_per_word"
        )
    if candidate_pool_multiplier < 1:
        raise ValueError("candidate_pool_multiplier must be at least 1")
    if not skip_coherence_check:
        _gemini_api_key()
    if checkpoint_path:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        if resume:
            samples = load_existing_samples(checkpoint_path, contexts_per_word)
            with checkpoint_path.open("w", encoding="utf-8") as handle:
                for sample in samples:
                    handle.write(json.dumps(asdict(sample), ensure_ascii=False) + "\n")
        else:
            checkpoint_path.write_text("", encoding="utf-8")
    if coherence_log_path:
        coherence_log_path.parent.mkdir(parents=True, exist_ok=True)
        if not resume:
            coherence_log_path.write_text("", encoding="utf-8")
        elif not coherence_log_path.exists():
            coherence_log_path.touch()
    completed_words = {sample.word for sample in samples}
    if sample_size > 0 and len(completed_words) >= sample_size:
        return samples, missing
    occurrence_index, corpus_char_count = index_corpus_occurrences(
        words={entry.word for entry in selected},
        corpus_files=corpus_files,
        context_tokens=context_tokens,
        exclude_capitalized_matches=exclude_capitalized_matches,
        max_occurrences_per_word=candidate_contexts_per_word,
        sampling_seed=seed,
    )
    raw_context_missing = [
        entry
        for entry in selected
        if len(occurrence_index.get(entry.word, [])) < contexts_per_word
    ]
    missing.extend(raw_context_missing)
    selected = [
        entry
        for entry in selected
        if len(occurrence_index.get(entry.word, [])) >= contexts_per_word
    ]
    selected = [entry for entry in selected if entry.word not in completed_words]
    if progress_interval > 0:
        print(
            f"Raw context filter kept {len(selected)} words "
            f"(skipped {len(raw_context_missing)} with fewer than {contexts_per_word} contexts)",
            flush=True,
        )

    words_sampled = len(completed_words)
    if progress_interval > 0 and completed_words:
        print(
            f"Resuming from {len(completed_words)} words / {len(samples)} samples",
            flush=True,
        )
    jobs: list[tuple[FrequencyEntry, int, list[IndexedOccurrence]]] = []
    for entry in selected:
        search_start = rng.randrange(corpus_char_count) if corpus_char_count > 0 else 0
        jobs.append(
            (
                entry,
                search_start,
                _ordered_occurrences(occurrence_index.get(entry.word, []), search_start),
            )
        )

    def validate_job(
        job: tuple[FrequencyEntry, int, list[IndexedOccurrence]],
    ) -> tuple[
        FrequencyEntry,
        int,
        list[IndexedOccurrence],
        int,
        int,
        list[IndexedOccurrence],
        list[ContextDecision],
    ]:
        entry, search_start, occurrences = job
        candidates = occurrences[:candidate_contexts_per_word]
        if skip_coherence_check:
            decisions = [
                ContextDecision(True, "accepted", "Coherence check skipped.")
                for _candidate in candidates
            ]
        elif coherence_log_path:
            decisions = validate_contexts_with_gemini_detailed(
                [occurrence.raw_excerpt for occurrence in candidates],
                target_word=entry.word,
                model=coherence_model,
                language=normalize_language(language),
            )
        else:
            accepted = validate_contexts_with_gemini(
                [occurrence.raw_excerpt for occurrence in candidates],
                target_word=entry.word,
                model=coherence_model,
                language=normalize_language(language),
            )
            decisions = [
                ContextDecision(
                    decision,
                    "accepted" if decision else "other",
                    "Boolean-only coherence decision.",
                )
                for decision in accepted
            ]
        accepted_all = [
            occurrence
            for occurrence, decision in zip(candidates, decisions)
            if decision.accepted
        ]
        return (
            entry,
            search_start,
            accepted_all[:contexts_per_word],
            len(accepted_all),
            len(candidates),
            candidates,
            decisions,
        )

    chunk_size = 1 if skip_coherence_check else coherence_workers
    with ThreadPoolExecutor(max_workers=chunk_size) as executor:
        for chunk_start in range(0, len(jobs), chunk_size):
            if sample_size > 0 and words_sampled >= sample_size:
                break
            chunk = jobs[chunk_start : chunk_start + chunk_size]
            for (
                entry,
                search_start,
                accepted,
                accepted_count,
                candidate_count,
                candidates,
                decisions,
            ) in executor.map(validate_job, chunk):
                if sample_size > 0 and words_sampled >= sample_size:
                    break
                if coherence_log_path:
                    with coherence_log_path.open("a", encoding="utf-8") as handle:
                        for candidate_index, (occurrence, decision) in enumerate(
                            zip(candidates, decisions)
                        ):
                            row = {
                                "word": entry.word,
                                "rank": entry.rank,
                                "candidate_index": candidate_index,
                                "accepted": decision.accepted,
                                "reason": decision.reason,
                                "note": decision.note,
                                "excerpt": occurrence.raw_excerpt,
                                "source_path": occurrence.source_path,
                                "match_start_char": occurrence.match_start_char,
                                "match_end_char": occurrence.match_end_char,
                            }
                            handle.write(
                                json.dumps(row, ensure_ascii=False) + "\n"
                            )
                if len(accepted) < contexts_per_word:
                    missing.append(entry)
                    if progress_interval > 0:
                        print(
                            f"Rejected {entry.word!r}: Gemini accepted "
                            f"{accepted_count}/{candidate_count} contexts; "
                            f"need {contexts_per_word}.",
                            flush=True,
                        )
                    continue

                committed_samples: list[Sample] = []
                for occurrence in accepted:
                    sample = Sample(
                        id=f"sample-{len(samples) + 1:06d}",
                        word=entry.word,
                        rank=entry.rank,
                        count=entry.count,
                        prefix=occurrence.prefix,
                        matched_text=occurrence.matched_text,
                        source_path=occurrence.source_path,
                        match_start_char=occurrence.match_start_char,
                        match_end_char=occurrence.match_end_char,
                        context_token_count=occurrence.context_token_count,
                        search_start_char=search_start,
                        metadata={
                            "rank_min": rank_min,
                            "rank_max": rank_max,
                            "sample_size": sample_size,
                            "frequency_path": str(frequency_path),
                            "corpus_path": str(corpus_path),
                            "context_tokens": context_tokens,
                            "seed": seed,
                            "selection": "filled_from_rank_band",
                            "exclude_capitalized_matches": int(
                                exclude_capitalized_matches
                            ),
                            "min_word_length": min_word_length,
                            "dictionary": (
                                str(dictionary_path) if dictionary_path else ""
                            ),
                            "contexts_per_word": contexts_per_word,
                            "candidate_contexts_per_word": (
                                candidate_contexts_per_word
                            ),
                            "candidate_pool_multiplier": candidate_pool_multiplier,
                            "validate_target_words": int(validate_target_words),
                            "target_word_batch_size": target_word_batch_size,
                            "coherence_log": (
                                str(coherence_log_path) if coherence_log_path else ""
                            ),
                            "coherence_model": coherence_model,
                            "coherence_workers": coherence_workers,
                            "skip_coherence_check": int(skip_coherence_check),
                            "language": normalize_language(language),
                        },
                    )
                    samples.append(sample)
                    committed_samples.append(sample)
                if checkpoint_path:
                    with checkpoint_path.open("a", encoding="utf-8") as handle:
                        for sample in committed_samples:
                            handle.write(
                                json.dumps(asdict(sample), ensure_ascii=False) + "\n"
                            )
                words_sampled += 1
                if progress_interval > 0 and words_sampled % progress_interval == 0:
                    print(
                        f"Accepted {words_sampled} words / {len(samples)} samples "
                        f"(skipped {len(missing)} words)",
                        flush=True,
                    )

    return samples, missing


def write_jsonl(samples: Iterable[Sample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(asdict(sample), ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build WCS target/context samples.")
    parser.add_argument("--frequency", type=Path, required=True, help="Ranked frequency list.")
    parser.add_argument("--corpus", type=Path, required=True, help="Corpus file or directory of text files.")
    parser.add_argument("--output", type=Path, default=Path("data/processed/samples.jsonl"))
    parser.add_argument("--rank-min", type=int, default=10_000)
    parser.add_argument("--rank-max", type=int, default=40_000)
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--context-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--min-word-length",
        type=int,
        default=3,
        help="Minimum normalized word length to keep from the frequency list.",
    )
    parser.add_argument(
        "--dictionary",
        type=Path,
        default=None,
        help="Optional newline-delimited dictionary. Targets must appear in it.",
    )
    parser.add_argument(
        "--exclude-capitalized-matches",
        action="store_true",
        help="Skip corpus matches whose surface form starts with a capital letter.",
    )
    parser.add_argument(
        "--contexts-per-word",
        type=int,
        default=10,
        help="Number of different contexts to pick for each selected word.",
    )
    parser.add_argument(
        "--coherence-model",
        default=DEFAULT_GEMINI_MODEL,
        help="Gemini model used for batched context validation.",
    )
    parser.add_argument(
        "--coherence-workers",
        type=int,
        default=DEFAULT_COHERENCE_WORKERS,
        help="Maximum number of concurrent Gemini validation requests.",
    )
    parser.add_argument(
        "--candidate-contexts-per-word",
        type=int,
        default=DEFAULT_CANDIDATE_CONTEXTS_PER_WORD,
        help=(
            "Candidate contexts sent together for each word. Must be at least "
            "--contexts-per-word."
        ),
    )
    parser.add_argument(
        "--candidate-pool-multiplier",
        type=int,
        default=DEFAULT_CANDIDATE_POOL_MULTIPLIER,
        help=(
            "Randomly index this many times the requested word count, allowing "
            "replacement words when contexts are rejected."
        ),
    )
    parser.add_argument(
        "--validate-target-words",
        action="store_true",
        help=(
            "Use Gemini to reject nonwords, names, abbreviations, and foreign-only "
            "forms before searching the corpus."
        ),
    )
    parser.add_argument(
        "--target-word-batch-size",
        type=int,
        default=DEFAULT_TARGET_WORD_BATCH_SIZE,
        help="Number of candidate target words classified in each Gemini request.",
    )
    parser.add_argument(
        "--skip-coherence-check",
        action="store_true",
        help="Accept raw corpus contexts without language/coherence filtering.",
    )
    parser.add_argument(
        "--coherence-log",
        type=Path,
        default=None,
        help=(
            "Optional JSONL audit log containing every context decision, reason, "
            "explanation, excerpt, and source location."
        ),
    )
    parser.add_argument(
        "--language",
        default="English",
        help="Language name or code for the coherence prompt, for example English or Spanish.",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=1,
        help="Print progress after this many accepted words. Use 0 to disable.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append to an existing output file, preserving complete word groups.",
    )
    parser.add_argument(
        "--require-full-sample",
        action="store_true",
        help="Exit with an error if fewer than --sample-size complete words are built.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples, missing = build_samples(
        frequency_path=args.frequency,
        corpus_path=args.corpus,
        rank_min=args.rank_min,
        rank_max=args.rank_max,
        sample_size=args.sample_size,
        context_tokens=args.context_tokens,
        seed=args.seed,
        exclude_capitalized_matches=args.exclude_capitalized_matches,
        min_word_length=args.min_word_length,
        dictionary_path=args.dictionary,
        contexts_per_word=args.contexts_per_word,
        coherence_model=args.coherence_model,
        language=args.language,
        checkpoint_path=args.output,
        progress_interval=args.progress_interval,
        resume=args.resume,
        skip_coherence_check=args.skip_coherence_check,
        coherence_workers=args.coherence_workers,
        candidate_contexts_per_word=args.candidate_contexts_per_word,
        candidate_pool_multiplier=args.candidate_pool_multiplier,
        validate_target_words=args.validate_target_words,
        target_word_batch_size=args.target_word_batch_size,
        coherence_log_path=args.coherence_log,
    )
    write_jsonl(samples, args.output)
    accepted_words = len({sample.word for sample in samples})
    print(
        f"Wrote {len(samples)} samples for {accepted_words} words to {args.output}"
    )
    if missing:
        print(f"Skipped {len(missing)} words with no full-context match")
    if (
        args.require_full_sample
        and args.sample_size > 0
        and accepted_words < args.sample_size
    ):
        raise SystemExit(
            f"Built {accepted_words}/{args.sample_size} required words. "
            "Increase --candidate-pool-multiplier, --candidate-contexts-per-word, "
            "or the context corpus size, then rerun with --resume."
        )


if __name__ == "__main__":
    main()
