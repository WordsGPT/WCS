"""Tokenization and length-robust lexical-diversity metrics."""

from __future__ import annotations

import re
from collections.abc import Iterable


WORD_RE = re.compile(r"[^\W\d_]+(?:['’-][^\W\d_]+)*", re.UNICODE)


def lexical_tokens(text: str) -> list[str]:
    """Return case-folded word tokens while retaining internal apostrophes/hyphens."""
    return [match.group(0).casefold() for match in WORD_RE.finditer(text)]


def type_token_ratio(tokens: Iterable[str]) -> float:
    values = list(tokens)
    return len(set(values)) / len(values) if values else 0.0


def _mtld_one_direction(tokens: list[str], threshold: float) -> float:
    if not tokens:
        return 0.0
    factors = 0.0
    types: set[str] = set()
    segment_length = 0
    for token in tokens:
        segment_length += 1
        types.add(token)
        ttr = len(types) / segment_length
        if ttr <= threshold:
            factors += 1.0
            types.clear()
            segment_length = 0
    if segment_length:
        ttr = len(types) / segment_length
        denominator = 1.0 - threshold
        factors += (1.0 - ttr) / denominator if denominator else 0.0
    return len(tokens) / factors if factors > 0 else float(len(tokens))


def mtld(tokens: Iterable[str], threshold: float = 0.72, bidirectional: bool = False) -> float:
    """Compute MTLD in forward order, matching ``LexicalRichness.mtld``.

    The partial-factor calculation follows the standard McCarthy/Jarvis MTLD
    definition. Set ``bidirectional`` for the forward/reverse mean variant.
    Empty inputs return zero.
    """
    values = list(tokens)
    if not 0.0 < threshold < 1.0:
        raise ValueError("MTLD threshold must be between 0 and 1")
    forward = _mtld_one_direction(values, threshold)
    if not bidirectional or not values:
        return forward
    reverse = _mtld_one_direction(list(reversed(values)), threshold)
    return (forward + reverse) / 2.0
