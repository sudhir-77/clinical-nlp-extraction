from __future__ import annotations

from pathlib import Path

from clinical_nlp.bioscope import parse_bioscope_abstracts
from clinical_nlp.negation import (
    is_dev_document,
    mention_flags,
    negation_scopes,
    overlaps,
    uncertainty_scopes,
)

FIXTURE = """<?xml version="1.0"?>
<Annotation>
  <DocumentSet>
    <Document type="Biological_abstract">
      <DocID type="PMID">42</DocID>
      <DocumentPart type="AbstractText">
        <sentence id="S1.1">There is <xcope id="X1"><cue type="negation" ref="X1">no</cue> evidence of cancer</xcope>.</sentence>
        <sentence id="S1.2">Results <xcope id="X2"><cue type="speculation" ref="X2">suggest</cue> a possible role</xcope>.</sentence>
      </DocumentPart>
    </Document>
  </DocumentSet>
</Annotation>
"""


def test_bioscope_scope_offsets(tmp_path: Path):
    path = tmp_path / "abstracts.xml"
    path.write_text(FIXTURE, encoding="utf-8")
    rows = parse_bioscope_abstracts(path)
    assert len(rows) == 2
    first = rows[0]
    assert first.text.startswith("There is no evidence")
    assert first.negation_spans
    start, end = first.negation_spans[0]
    assert "no evidence of cancer" in first.text[start:end]
    assert rows[1].speculation_spans
    assert not rows[1].negation_spans


def test_negex_known_cases():
    negated = "There is no evidence of pneumonia today."
    scopes = negation_scopes(negated)
    start = negated.index("pneumonia")
    assert any(overlaps(start, start + len("pneumonia"), a, b) for a, b in scopes)

    affirmed = "Pneumonia was confirmed today."
    start = affirmed.index("Pneumonia")
    assert not any(
        overlaps(start, start + len("Pneumonia"), a, b) for a, b in negation_scopes(affirmed)
    )

    pseudo = "Not only pneumonia was found."
    start = pseudo.index("pneumonia")
    assert not any(
        overlaps(start, start + len("pneumonia"), a, b) for a, b in negation_scopes(pseudo)
    )

    asthma = "Asthma can not be excluded."
    start = asthma.index("Asthma")
    assert any(overlaps(start, start + len("Asthma"), a, b) for a, b in negation_scopes(asthma))

    lymphoma = "non-Hodgkin lymphoma is present."
    assert negation_scopes(lymphoma) == []

    hedged = "Possible pneumonia was noted."
    start = hedged.index("pneumonia")
    negated_flag, uncertain = mention_flags(hedged, start, start + len("pneumonia"))
    assert not negated_flag and uncertain
    assert uncertainty_scopes(hedged)


def test_dev_slice_is_stable_and_partitions():
    assert is_dev_document("42") == is_dev_document("42")
    flags = {is_dev_document(str(i)) for i in range(30)}
    assert flags == {True, False}
