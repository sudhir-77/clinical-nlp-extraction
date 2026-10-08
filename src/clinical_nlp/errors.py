"""Categorize exact-span NER mistakes on the official test predictions."""

from __future__ import annotations

from clinical_nlp.metrics import spans_from_tags


def _overlap(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return left[0] < right[1] and right[0] < left[1]


def errors_in_sentence(gold_tags: list[str], pred_tags: list[str]) -> list[dict]:
    gold = spans_from_tags(gold_tags)
    pred = spans_from_tags(pred_tags)
    matched_gold = set()
    matched_pred = set()
    for gold_index, gold_span in enumerate(gold):
        for pred_index, pred_span in enumerate(pred):
            if gold_span == pred_span:
                matched_gold.add(gold_index)
                matched_pred.add(pred_index)
    found: list[dict] = []
    for gold_index, gold_span in enumerate(gold):
        if gold_index in matched_gold:
            continue
        partners = [span for span in pred if _overlap(gold_span, span)]
        if partners:
            found.append(
                {
                    "category": "boundary",
                    "gold_span": list(gold_span),
                    "pred_span": list(partners[0]),
                }
            )
        else:
            found.append(
                {
                    "category": "missed",
                    "gold_span": list(gold_span),
                    "pred_span": None,
                }
            )
    for pred_index, pred_span in enumerate(pred):
        if pred_index in matched_pred:
            continue
        if any(_overlap(pred_span, span) for span in gold):
            continue
        found.append(
            {
                "category": "false_positive",
                "gold_span": None,
                "pred_span": list(pred_span),
            }
        )
    return found


def select_errors(records: list[dict], limit: int = 50) -> list[dict]:
    buckets = {"boundary": [], "missed": [], "false_positive": []}
    for record in records:
        buckets[record["category"]].append(record)
    selected: list[dict] = []
    while len(selected) < limit and any(buckets.values()):
        for category in ("boundary", "missed", "false_positive"):
            if buckets[category] and len(selected) < limit:
                selected.append(buckets[category].pop(0))
    return selected


def surface(tokens: list[str], span: list[int] | None) -> str | None:
    if span is None:
        return None
    return " ".join(tokens[span[0] : span[1]])


def analyze(test_rows: list[dict], limit: int = 50) -> dict:
    detailed: list[dict] = []
    for row in test_rows:
        for error in errors_in_sentence(row["gold_tags"], row["pred_tags"]):
            detailed.append(
                {
                    "sentence_id": row["sentence_id"],
                    "category": error["category"],
                    "tokens": row["tokens"],
                    "gold_text": surface(row["tokens"], error["gold_span"]),
                    "pred_text": surface(row["tokens"], error["pred_span"]),
                }
            )
    counts = {
        "boundary": sum(1 for row in detailed if row["category"] == "boundary"),
        "missed": sum(1 for row in detailed if row["category"] == "missed"),
        "false_positive": sum(1 for row in detailed if row["category"] == "false_positive"),
    }
    chosen = select_errors(detailed, limit)
    return {
        "n_errors": len(detailed),
        "n_analyzed": len(chosen),
        "counts": counts,
        "examples": [
            {
                "sentence_id": row["sentence_id"],
                "category": row["category"],
                "gold_text": row["gold_text"],
                "pred_text": row["pred_text"],
            }
            for row in chosen
        ],
    }
