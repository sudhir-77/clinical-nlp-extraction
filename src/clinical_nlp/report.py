"""Write metrics.json, the README results table, and resume notes from code."""

from __future__ import annotations

import json
from pathlib import Path


REQUIRED_TOP_KEYS = (
    "project",
    "success_criteria",
    "metrics",
    "baseline",
    "splits",
    "seeds",
    "hardware",
    "runtime_seconds",
)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def criterion(name: str, value: float, op: str, threshold: float) -> dict:
    if op == ">=":
        passed = value >= threshold
    elif op == ">":
        passed = value > threshold
    elif op == "<=":
        passed = value <= threshold
    elif op == "<":
        passed = value < threshold
    else:
        raise ValueError(op)
    return {
        "name": name,
        "value": value,
        "op": op,
        "threshold": threshold,
        "threshold_high": None,
        "passed": bool(passed),
    }


def render_results_table(payload: dict) -> str:
    lines = [
        "| Metric | Value | Threshold | Passed |",
        "| --- | --- | --- | --- |",
    ]
    for row in payload["success_criteria"]:
        lines.append(
            f"| {row['name']} | {row['value']:.4f} | {row['op']} {row['threshold']} | {row['passed']} |"
        )
    ner = payload["metrics"]["ncbi_disease"]
    lines.append("")
    lines.append(
        f"Transformer entity F1 {ner['transformer']['f1']:.4f} "
        f"(precision {ner['transformer']['precision']:.4f}, "
        f"recall {ner['transformer']['recall']:.4f}). "
        f"CRF entity F1 {ner['crf']['f1']:.4f}. "
        f"Negation F1 {payload['metrics']['bioscope']['held_out']['f1']:.4f}."
    )
    return "\n".join(lines)


def update_readme(readme: Path, payload: dict) -> None:
    text = readme.read_text(encoding="utf-8")
    start = "<!-- RESULTS_TABLE_START -->"
    end = "<!-- RESULTS_TABLE_END -->"
    block = f"{start}\n{render_results_table(payload)}\n{end}"
    if start in text and end in text:
        before, rest = text.split(start, 1)
        _old, after = rest.split(end, 1)
        text = before + block + after
    else:
        text = text.rstrip() + "\n\n" + block + "\n"
    readme.write_text(text, encoding="utf-8")


def _round(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def write_resume_notes(path: Path, payload: dict) -> None:
    ner = payload["metrics"]["ncbi_disease"]
    neg = payload["metrics"]["bioscope"]["held_out"]["f1"]
    api_percent = payload["metrics"]["api"]["valid_percent"]
    analyzed = payload["metrics"]["error_analysis"]["n_analyzed"]
    f1 = _round(ner["transformer"]["f1"])
    base = _round(ner["crf"]["f1"])
    neg_text = _round(neg)
    bullets = [
        (
            f"Fine-tuned a biomedical transformer for disease NER on 793 expert-annotated "
            f"abstracts, reaching {f1} entity-level F1 vs. {base} for a CRF baseline"
        ),
        (
            f"Added NegEx-style negation and uncertainty detection ({neg_text} F1) and "
            f"served extraction as a FastAPI endpoint with {api_percent:.0f}% schema-valid JSON"
        ),
        (
            f"Analyzed {analyzed} test errors by boundary, miss and false-positive type "
            f"to guide the next labeling pass"
        ),
    ]
    summary = (
        "This project fine-tunes a biomedical transformer to find disease mentions in "
        f"PubMed abstracts and compares it with a dictionary-plus-CRF baseline "
        f"({f1} vs. {base} entity F1 on the official NCBI Disease test split). "
        f"A NegEx-style module labels negation ({neg_text} F1 on held-out BioScope "
        "abstract sentences) and the same pipeline is served as JSON from FastAPI."
    )
    text = "\n".join(
        [
            "# Resume notes",
            "",
            "## Bullets",
            "",
            *[f"- {bullet}" for bullet in bullets],
            "",
            "## Skills and tools",
            "",
            "Python, PyTorch, Hugging Face Transformers, NCBI Disease corpus, BioScope, "
            "sklearn-crfsuite, seqeval, FastAPI, pydantic, pytest",
            "",
            "## Summary",
            "",
            summary,
            "",
            "## GitHub description",
            "",
            "Fine-tune a biomedical transformer for disease NER, compare it with a dictionary CRF, "
            "and serve NegEx-style extraction as a FastAPI endpoint.",
            "",
        ]
    )
    path.write_text(text, encoding="utf-8")
