from fastapi.testclient import TestClient

from clinical_nlp.api import create_app
from clinical_nlp.errors import analyze, errors_in_sentence
from clinical_nlp.pipeline import ExtractionPipeline
from clinical_nlp.schema import ExtractResponse


class StubTagger:
    def predict_tags(self, tokens: list[str]) -> list[str]:
        tags = []
        for token in tokens:
            if token.lower() in {"cancer", "pneumonia"}:
                tags.append("B-DISEASE")
            else:
                tags.append("O")
        return tags


def test_extract_endpoint_schema_and_negation():
    app = create_app(ExtractionPipeline(StubTagger()))
    with TestClient(app) as client:
        empty = client.post("/extract", json={"text": ""})
        assert empty.status_code == 422
        negated = client.post("/extract", json={"text": "There is no evidence of cancer."})
        assert negated.status_code == 200
        body = ExtractResponse.model_validate(negated.json())
        assert len(body.conditions) == 1
        assert body.conditions[0].condition == "cancer"
        assert body.conditions[0].negated is True
        assert body.conditions[0].uncertain is False
        hedged = client.post("/extract", json={"text": "Possible pneumonia was noted."})
        hedged_body = ExtractResponse.model_validate(hedged.json())
        assert hedged_body.conditions[0].uncertain is True
        assert hedged_body.conditions[0].negated is False
        plain = client.post("/extract", json={"text": "The patient has cancer."})
        plain_body = ExtractResponse.model_validate(plain.json())
        assert plain_body.conditions[0].negated is False


def test_error_categories_and_sample_of_fifty():
    boundary = errors_in_sentence(
        ["B-DISEASE", "I-DISEASE", "O"],
        ["B-DISEASE", "O", "O"],
    )
    assert [row["category"] for row in boundary] == ["boundary"]
    missed = errors_in_sentence(["B-DISEASE", "O"], ["O", "O"])
    assert missed[0]["category"] == "missed"
    false_positive = errors_in_sentence(["O", "O"], ["B-DISEASE", "O"])
    assert false_positive[0]["category"] == "false_positive"
    rows = []
    for index in range(40):
        rows.append(
            {
                "sentence_id": f"s{index}",
                "tokens": ["cancer", "today"],
                "gold_tags": ["B-DISEASE", "O"],
                "pred_tags": ["O", "O"],
            }
        )
    for index in range(20):
        rows.append(
            {
                "sentence_id": f"b{index}",
                "tokens": ["colon", "cancer"],
                "gold_tags": ["B-DISEASE", "I-DISEASE"],
                "pred_tags": ["B-DISEASE", "O"],
            }
        )
    report = analyze(rows, limit=50)
    assert report["n_analyzed"] == 50
    assert report["counts"]["missed"] == 40
    assert report["counts"]["boundary"] == 20
    categories = {row["category"] for row in report["examples"]}
    assert "missed" in categories and "boundary" in categories
