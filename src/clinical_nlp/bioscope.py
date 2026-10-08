"""BioScope abstracts: negation and speculation scopes. Clinical reports are not read."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class BioSentence:
    sentence_id: str
    doc_id: str
    text: str
    negation_spans: list[tuple[int, int]] = field(default_factory=list)
    speculation_spans: list[tuple[int, int]] = field(default_factory=list)


def _scope_type(element: ET.Element) -> str | None:
    scope_id = element.attrib.get("id")
    for cue in element.iter("cue"):
        if cue.attrib.get("ref") == scope_id and cue.attrib.get("type"):
            return cue.attrib["type"]
    for child in list(element):
        if child.tag == "cue" and child.attrib.get("type"):
            return child.attrib["type"]
    return None


def _sentence_text_and_scopes(
    sentence: ET.Element,
) -> tuple[str, list[tuple[int, int, str]]]:
    parts: list[str] = []
    located: list[tuple[int, int, ET.Element]] = []

    def walk(element: ET.Element) -> None:
        if element.text:
            parts.append(element.text)
        for child in list(element):
            if child.tag == "xcope":
                start = sum(len(part) for part in parts)
                walk(child)
                end = sum(len(part) for part in parts)
                located.append((start, end, child))
            else:
                walk(child)
            if child.tail:
                parts.append(child.tail)

    walk(sentence)
    scopes: list[tuple[int, int, str]] = []
    for start, end, element in located:
        kind = _scope_type(element)
        if kind and end > start:
            scopes.append((start, end, kind))
    return "".join(parts), scopes


def parse_bioscope_abstracts(path: Path) -> list[BioSentence]:
    root = ET.parse(path).getroot()
    rows: list[BioSentence] = []
    for document in root.iter("Document"):
        doc_id = ""
        for child in list(document):
            if child.tag == "DocID":
                doc_id = "".join(child.itertext()).strip()
                break
        if not doc_id:
            doc_id = f"doc{len(rows)}"
        for sentence in document.iter("sentence"):
            text, scopes = _sentence_text_and_scopes(sentence)
            if not text.strip():
                continue
            sid = sentence.attrib.get("id", str(len(rows)))
            rows.append(
                BioSentence(
                    sentence_id=f"{doc_id}:{sid}",
                    doc_id=doc_id,
                    text=text,
                    negation_spans=[(a, b) for a, b, kind in scopes if kind == "negation"],
                    speculation_spans=[
                        (a, b) for a, b, kind in scopes if kind == "speculation"
                    ],
                )
            )
    return rows


def count_documents(path: Path) -> int:
    root = ET.parse(path).getroot()
    return sum(1 for _ in root.iter("Document"))
