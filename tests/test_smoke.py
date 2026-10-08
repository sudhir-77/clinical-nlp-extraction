from pathlib import Path

from clinical_nlp.env import project_root
from clinical_nlp.runner import run_pipeline


def test_quick_pipeline_writes_artifacts(tmp_path: Path):
    payload = run_pipeline(project_root(), quick=True, results_dir=tmp_path)
    prediction = (tmp_path / "test_predictions.jsonl").read_text(encoding="utf-8").splitlines()
    assert prediction
    import json

    row = json.loads(prediction[0])
    for key in ("sentence_id", "tokens", "gold_tags", "pred_tags", "crf_tags"):
        assert key in row
    assert len(row["tokens"]) == len(row["gold_tags"]) == len(row["pred_tags"]) == len(row["crf_tags"])
    header = (tmp_path / "negation_predictions.csv").read_text(encoding="utf-8").splitlines()[0]
    assert header == "sentence_id,mention,gold_negated,pred_negated"
    api = json.loads((tmp_path / "api_check.json").read_text(encoding="utf-8"))
    assert api["n_notes"] == api["n_valid"] > 0
    assert payload["splits"]["ncbi_disease_abstracts"] == {"train": 593, "val": 100, "test": 100}
    held_out = payload["metrics"]["bioscope"]["held_out"]
    assert held_out["support"] >= 50
    assert held_out["f1"] >= 0.80
    assert payload["seeds"] == [0]
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "error_analysis.json").exists()
