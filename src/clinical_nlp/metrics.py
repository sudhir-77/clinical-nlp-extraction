"""Entity-level scores and BIO helpers. Test labels are not used here."""

from __future__ import annotations

from seqeval.metrics import f1_score as seqeval_f1
from seqeval.metrics import precision_score as seqeval_precision
from seqeval.metrics import recall_score as seqeval_recall
from sklearn.metrics import f1_score, precision_score, recall_score


def repair_bio(tags: list[str]) -> list[str]:
    """Turn illegal I- tags into B- so exact-span scoring stays well defined."""
    repaired: list[str] = []
    previous = "O"
    for tag in tags:
        if tag.startswith("I-"):
            entity = tag[2:]
            previous_entity = previous[2:] if previous != "O" else ""
            if previous == "O" or previous_entity != entity:
                tag = f"B-{entity}"
        repaired.append(tag)
        previous = tag
    return repaired


def spans_from_tags(tags: list[str]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    index = 0
    while index < len(tags):
        tag = tags[index]
        if tag.startswith("B-"):
            entity = tag[2:]
            end = index + 1
            while end < len(tags) and tags[end] == f"I-{entity}":
                end += 1
            spans.append((index, end))
            index = end
        else:
            index += 1
    return spans


def entity_scores(gold: list[list[str]], pred: list[list[str]]) -> dict[str, float]:
    if not gold:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    return {
        "precision": float(seqeval_precision(gold, pred)),
        "recall": float(seqeval_recall(gold, pred)),
        "f1": float(seqeval_f1(gold, pred)),
    }


def binary_scores(gold: list[int], pred: list[int]) -> dict[str, float]:
    if not gold:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0}
    return {
        "precision": float(precision_score(gold, pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(gold, pred, pos_label=1, zero_division=0)),
        "f1": float(f1_score(gold, pred, pos_label=1, zero_division=0)),
        "support": int(sum(gold)),
    }
