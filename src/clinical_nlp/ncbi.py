"""Parse the official NCBI Disease mention files into sentence-level BIO tags."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from clinical_nlp.tokenize import split_sentences, tokenize_with_spans

EXPECTED_COUNTS = {
    "train": {"documents": 593, "mentions": 5145},
    "dev": {"documents": 100, "mentions": 787},
    "test": {"documents": 100, "mentions": 960},
}


@dataclass
class Document:
    pmid: str
    doc_index: int
    title: str
    abstract: str
    mentions: list[tuple[int, int, str]] = field(default_factory=list)

    @property
    def text(self) -> str:
        if self.abstract:
            return f"{self.title} {self.abstract}"
        return self.title


def parse_ncbi_text(text: str) -> list[Document]:
    documents: list[Document] = []
    current: Document | None = None

    def close() -> None:
        nonlocal current
        if current is not None:
            documents.append(current)
            current = None

    for raw in text.splitlines():
        line = raw.strip("\n")
        if not line.strip():
            close()
            continue
        if "|t|" in line or "|a|" in line:
            pmid, kind, content = line.split("|", 2)
            if current is None or current.pmid != pmid:
                close()
                current = Document(
                    pmid=pmid,
                    doc_index=len(documents),
                    title="",
                    abstract="",
                )
            if kind == "t":
                current.title = content
            elif kind == "a":
                current.abstract = content
            else:
                raise ValueError(f"Unknown NCBI row kind {kind}")
            continue
        parts = line.split("\t")
        if len(parts) < 6:
            raise ValueError(f"Bad NCBI annotation row: {line[:80]}")
        if current is None or current.pmid != parts[0]:
            raise ValueError(f"Annotation before document header: {parts[0]}")
        start, end = int(parts[1]), int(parts[2])
        mention = parts[3]
        body = current.text
        if not (0 <= start < end <= len(body)):
            raise ValueError(f"Span out of range in {current.pmid}: {start}:{end}")
        spanned = body[start:end]
        if spanned != mention and spanned.strip() != mention.strip():
            raise ValueError(
                f"Span text mismatch in {current.pmid}: {mention!r} vs {spanned!r}"
            )
        current.mentions.append((start, end, spanned))
    close()
    return documents


def parse_ncbi_file(path: Path) -> list[Document]:
    return parse_ncbi_text(path.read_text(encoding="utf-8"))


def _assign_tags(
    token_spans: list[tuple[int, int]],
    mentions: list[tuple[int, int, str]],
) -> list[str]:
    tags = ["O"] * len(token_spans)
    ordered = sorted(mentions, key=lambda item: (item[0] - item[1], item[0]))
    for start, end, _mention in ordered:
        covered = [
            i
            for i, (tok_start, tok_end) in enumerate(token_spans)
            if tok_start < end and tok_end > start
        ]
        if not covered or any(tags[i] != "O" for i in covered):
            continue
        tags[covered[0]] = "B-DISEASE"
        for index in covered[1:]:
            tags[index] = "I-DISEASE"
    return tags


def document_to_sentences(document: Document, split: str) -> list[dict]:
    """Sentence records. Mention character offsets are protected from splits."""
    protected = [(start, end) for start, end, _mention in document.mentions]
    sentences = []
    for sent_index, (sent_start, sent_end) in enumerate(
        split_sentences(document.text, protected)
    ):
        sentence = document.text[sent_start:sent_end]
        tokens, local_spans = tokenize_with_spans(sentence)
        if not tokens:
            continue
        for token, (start, end) in zip(tokens, local_spans):
            if sentence[start:end] != token:
                raise ValueError(f"Token alignment failed in {document.pmid}")
        absolute = [(sent_start + a, sent_start + b) for a, b in local_spans]
        local_mentions = [
            (start, end, text)
            for start, end, text in document.mentions
            if start < sent_end and end > sent_start
        ]
        tags = _assign_tags(absolute, local_mentions)
        sentences.append(
            {
                "sentence_id": f"{document.pmid}:{document.doc_index}:{sent_index}",
                "pmid": document.pmid,
                "doc_index": document.doc_index,
                "split": split,
                "text": sentence,
                "tokens": tokens,
                "tags": tags,
                "token_spans": [list(span) for span in local_spans],
            }
        )
    return sentences


def sentences_from_documents(documents: list[Document], split: str) -> list[dict]:
    rows: list[dict] = []
    for document in documents:
        rows.extend(document_to_sentences(document, split))
    return rows


def cross_split_pmids(by_split: dict[str, list[Document]]) -> set[str]:
    grouped = {name: {doc.pmid for doc in docs} for name, docs in by_split.items()}
    overlap: set[str] = set()
    names = list(grouped)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            overlap |= grouped[left] & grouped[right]
    return overlap


def assert_official_counts(by_split: dict[str, list[Document]]) -> None:
    overlap = cross_split_pmids(by_split)
    if overlap:
        raise RuntimeError(f"PMID leakage across splits: {sorted(overlap)[:5]}")
    for split, expected in EXPECTED_COUNTS.items():
        documents = by_split[split]
        mentions = sum(len(doc.mentions) for doc in documents)
        if len(documents) != expected["documents"] or mentions != expected["mentions"]:
            raise RuntimeError(
                f"{split} counts {len(documents)} docs / {mentions} mentions, "
                f"expected {expected['documents']} / {expected['mentions']}"
            )
