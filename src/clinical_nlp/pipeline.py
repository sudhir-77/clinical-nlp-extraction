"""Turn a note into conditions with negation and uncertainty flags."""

from __future__ import annotations

from typing import Protocol

from clinical_nlp.metrics import repair_bio, spans_from_tags
from clinical_nlp.negation import mention_flags
from clinical_nlp.schema import Condition, ExtractResponse
from clinical_nlp.tokenize import split_sentences, tokenize_with_spans


class Tagger(Protocol):
    def predict_tags(self, tokens: list[str]) -> list[str]:
        """Return BIO tags aligned to tokens."""


class ExtractionPipeline:
    def __init__(self, tagger: Tagger) -> None:
        self.tagger = tagger

    def extract(self, text: str) -> ExtractResponse:
        conditions: list[Condition] = []
        for sent_start, sent_end in split_sentences(text):
            sentence = text[sent_start:sent_end]
            tokens, spans = tokenize_with_spans(sentence)
            if not tokens:
                continue
            tags = repair_bio(self.tagger.predict_tags(tokens))
            if len(tags) != len(tokens):
                raise RuntimeError("Tagger returned a different number of tags than tokens")
            for start_index, end_index in spans_from_tags(tags):
                start = sent_start + spans[start_index][0]
                end = sent_start + spans[end_index - 1][1]
                surface = text[start:end]
                if not surface.strip():
                    continue
                negated, uncertain = mention_flags(text, start, end)
                conditions.append(
                    Condition(
                        condition=surface,
                        span=(start, end),
                        negated=negated,
                        uncertain=uncertain,
                    )
                )
        return ExtractResponse(conditions=conditions)
