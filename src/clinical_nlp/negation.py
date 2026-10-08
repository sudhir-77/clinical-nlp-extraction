"""NegEx/ConText style negation and uncertainty scopes.

Cue list and window were frozen on a document-level development slice of
BioScope abstracts: zlib.crc32(pmid) % 10 < 3. The reported negation F1 uses
the complementary documents only.
"""

from __future__ import annotations

import zlib

from clinical_nlp.tokenize import tokenize_with_spans

# High-precision cues. Rare or ambiguous strings (fail, negative, except, or)
# are omitted because BioScope often does not treat them as negation cues.
# Forward window and lookback were chosen by F1 on the dev document slice only.
UNIGRAM_CUES = {
    "not", "no", "without", "absence", "lack", "lacking", "lacks", "lacked",
    "neither", "nor", "unable", "cannot", "absent", "never", "none",
}
PHRASE_CUES = (
    ("with", "the", "exception", "of"),
    ("rather", "than"),
    ("instead", "of"),
    ("no", "longer"),
    ("could", "not"),
    ("can", "not"),
)
PSEUDO_CUES = (
    ("not", "only"),
    ("not", "just"),
    ("not", "merely"),
    ("not", "simply"),
)
UNCERTAINTY_UNIGRAMS = {
    "possible", "possibly", "probable", "probably", "likely", "unlikely",
    "suspected", "suspicion", "suggestive", "potential", "potentially",
    "hypothesize", "hypothesized", "hypothesis", "appears", "apparently",
    "unclear", "unknown", "uncertain", "putative", "presumably", "presumed",
    "may", "might",
}
BOUNDARIES = {",", ";", ".", ":", "but", "however", "although", "whereas"}
FORWARD_WINDOW = 15
LOOKBACK = 1
DEV_MODULUS = 10
DEV_CUTOFF = 3


def is_dev_document(doc_id: str) -> bool:
    digest = zlib.crc32(doc_id.encode("utf-8")) & 0xFFFFFFFF
    return digest % DEV_MODULUS < DEV_CUTOFF


def _find_cues(
    words: list[str],
    phrases: tuple[tuple[str, ...], ...],
    unigrams: set[str],
) -> list[tuple[int, int]]:
    blocked = [False] * len(words)
    index = 0
    while index < len(words):
        matched = False
        for phrase in PSEUDO_CUES:
            length = len(phrase)
            if tuple(words[index : index + length]) == phrase:
                for cursor in range(index, index + length):
                    blocked[cursor] = True
                index += length
                matched = True
                break
        if not matched:
            index += 1
    ordered = sorted(phrases, key=len, reverse=True)
    cues: list[tuple[int, int]] = []
    index = 0
    while index < len(words):
        if blocked[index]:
            index += 1
            continue
        length = 0
        for phrase in ordered:
            size = len(phrase)
            if (
                tuple(words[index : index + size]) == phrase
                and not any(blocked[index : index + size])
            ):
                length = size
                break
        if length == 0 and words[index] in unigrams:
            length = 1
        if length:
            cues.append((index, index + length))
            index += length
        else:
            index += 1
    return cues


def _scopes_for(
    text: str,
    phrases: tuple[tuple[str, ...], ...],
    unigrams: set[str],
    window: int = FORWARD_WINDOW,
    lookback: int = LOOKBACK,
) -> list[tuple[int, int]]:
    tokens, spans = tokenize_with_spans(text)
    words = [token.lower() for token in tokens]
    cues = _find_cues(words, phrases, unigrams)
    scopes: list[tuple[int, int]] = []
    count = len(words)
    for start, end in cues:
        left = start
        if lookback:
            cursor = start - 1
            steps = 0
            while cursor >= 0 and words[cursor] not in BOUNDARIES and steps < lookback:
                cursor -= 1
                steps += 1
            left = cursor + 1
        cursor = end
        steps = 0
        while cursor < count and words[cursor] not in BOUNDARIES and steps < window:
            cursor += 1
            steps += 1
        if cursor <= left:
            continue
        scopes.append((spans[left][0], spans[cursor - 1][1]))
    return scopes


def negation_scopes(text: str) -> list[tuple[int, int]]:
    return _scopes_for(text, PHRASE_CUES, UNIGRAM_CUES)


def uncertainty_scopes(text: str) -> list[tuple[int, int]]:
    return _scopes_for(text, (), UNCERTAINTY_UNIGRAMS, window=15, lookback=2)


def overlaps(start: int, end: int, other_start: int, other_end: int) -> bool:
    return start < other_end and other_start < end


def mention_flags(text: str, start: int, end: int) -> tuple[bool, bool]:
    negated = any(overlaps(start, end, left, right) for left, right in negation_scopes(text))
    uncertain = any(
        overlaps(start, end, left, right) for left, right in uncertainty_scopes(text)
    )
    return negated, uncertain
