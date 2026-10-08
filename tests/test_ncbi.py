from __future__ import annotations

from pathlib import Path

import pytest

from clinical_nlp.ncbi import (
    assert_official_counts,
    cross_split_pmids,
    document_to_sentences,
    parse_ncbi_text,
)
from clinical_nlp.tokenize import split_sentences


def _corpus() -> str:
    title = "Alpha cancer study."
    abstract = "Patients had colon cancer."
    body = f"{title} {abstract}"
    cancer = body.index("cancer")
    colon = body.index("colon cancer")
    lines = [
        f"1|t|{title}",
        f"1|a|{abstract}",
        f"1\t{cancer}\t{cancer + len('cancer')}\tcancer\tSpecificDisease\tD1",
        f"1\t{colon}\t{colon + len('colon cancer')}\tcolon cancer\tSpecificDisease\tD2",
        "",
        "2|t|Methods only.",
        "2|a|No annotated mention.",
        "",
    ]
    return "\n".join(lines)


def test_offsets_match_and_sentences_keep_mentions():
    documents = parse_ncbi_text(_corpus())
    assert len(documents) == 2
    assert documents[0].mentions[1][2] == "colon cancer"
    rows = document_to_sentences(documents[0], "train")
    flat = " ".join(token for row in rows for token in row["tokens"])
    assert "colon" in flat and "cancer" in flat
    tagged = [tag for row in rows for tag in row["tags"]]
    assert tagged.count("B-DISEASE") == 2
    assert "I-DISEASE" in tagged
    for row in rows:
        for token, span in zip(row["tokens"], row["token_spans"]):
            assert row["text"][span[0] : span[1]] == token


def test_cross_split_pmids_and_official_guard():
    docs = parse_ncbi_text(_corpus())
    by_split = {"train": docs[:1], "dev": docs[1:], "test": []}
    assert cross_split_pmids(by_split) == set()
    leaked = {"train": docs, "dev": docs, "test": []}
    assert cross_split_pmids(leaked) == {"1", "2"}
    with pytest.raises(RuntimeError):
        assert_official_counts({"train": docs, "dev": [], "test": []})


def test_abbreviation_and_protected_spans():
    parts = split_sentences("See Fig. 1. Cancer is real.")
    assert len(parts) == 2
    text = "Alpha. Beta disease. Next sentence."
    assert len(split_sentences(text)) == 3
    end = text.index("disease") + len("disease")
    protected = split_sentences(text, [(0, end)])
    assert len(protected) == 2
    assert text[protected[0][0] : protected[0][1]].startswith("Alpha. Beta")


def test_duplicate_pmid_gets_distinct_sentence_ids():
    text = "9|t|First cancer.\n9|a|More cancer.\n9\t6\t12\tcancer\tSpecificDisease\tD1\n\n9|t|First cancer.\n9|a|More cancer.\n9\t6\t12\tcancer\tSpecificDisease\tD1\n"
    docs = parse_ncbi_text(text)
    assert len(docs) == 2
    assert docs[0].doc_index != docs[1].doc_index
    left = document_to_sentences(docs[0], "train")
    right = document_to_sentences(docs[1], "train")
    assert left[0]["sentence_id"] != right[0]["sentence_id"]
