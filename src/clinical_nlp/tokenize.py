"""Word tokenizer shared by the CRF, the transformer, and NegEx."""

from __future__ import annotations

import re

# Internal hyphens stay inside a token so "non-Hodgkin" is not read as negation.
TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*|[^\sA-Za-z0-9]")

_ABBREV = {
    "dr", "mr", "mrs", "ms", "prof", "fig", "figs", "eq", "al", "vs", "etc",
    "approx", "ca", "vol", "dept", "st", "jr", "sr", "inc", "ltd", "no", "nos",
    "ref", "refs", "cf", "viz", "eg", "ie",
}


def tokenize_with_spans(text: str) -> tuple[list[str], list[tuple[int, int]]]:
    tokens: list[str] = []
    spans: list[tuple[int, int]] = []
    for match in TOKEN_RE.finditer(text):
        tokens.append(match.group())
        spans.append((match.start(), match.end()))
    return tokens, spans


def split_sentences(
    text: str,
    protected: list[tuple[int, int]] | None = None,
) -> list[tuple[int, int]]:
    """Split on .?! while keeping abbreviations and protected mention spans intact."""
    blocked = protected or []
    spans: list[tuple[int, int]] = []
    start = 0
    i = 0
    n = len(text)
    while i < n:
        if text[i] in ".!?" and not any(s <= i < e for s, e in blocked):
            j = i - 1
            while j >= start and text[j].isalpha():
                j -= 1
            word = text[j + 1 : i].lower()
            if word in _ABBREV or len(word) == 1:
                i += 1
                continue
            if i > 0 and i + 1 < n and text[i - 1].isdigit() and text[i + 1].isdigit():
                i += 1
                continue
            end = i + 1
            while end < n and text[end] in "\"'”’":
                end += 1
            if text[start:end].strip():
                spans.append((start, end))
            k = end
            while k < n and text[k].isspace():
                k += 1
            start = k
            i = k
            continue
        i += 1
    if start < n and text[start:].strip():
        spans.append((start, n))
    return spans
