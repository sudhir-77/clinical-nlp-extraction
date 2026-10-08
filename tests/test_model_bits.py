from pathlib import Path

from clinical_nlp.download import sha256_bytes
from clinical_nlp.ner import (
    LABELS,
    build_quick_model,
    build_quick_tokenizer,
    encode_example,
    set_trainable,
)
from clinical_nlp.plots import write_plots
from clinical_nlp.report import criterion, write_resume_notes


def test_sha256_known_value():
    assert (
        sha256_bytes(b"abc")
        == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )


def test_subword_labels_use_first_piece_only(tmp_path: Path):
    vocab = tmp_path / "vocab.txt"
    vocab.write_text(
        "\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "car", "##cino", "##ma", "cancer"])
        + "\n",
        encoding="utf-8",
    )
    from transformers import BertTokenizerFast

    tokenizer = BertTokenizerFast(vocab_file=str(vocab), do_lower_case=True)
    ids, labels = encode_example(["carcinoma"], ["B-DISEASE"], tokenizer, max_length=16)
    pieces = tokenizer.convert_ids_to_tokens(ids)
    assert pieces[1:4] == ["car", "##cino", "##ma"]
    assert labels[1] == 1
    assert labels[2] == -100 and labels[3] == -100


def test_frozen_layers_keep_classifier_trainable(tmp_path: Path):
    sentences = [{"tokens": ["cancer", "risk"], "tags": ["B-DISEASE", "O"]}]
    tokenizer = build_quick_tokenizer(sentences, tmp_path / "vocab.txt")
    model = build_quick_model(tokenizer)
    set_trainable(model, n_unfreeze=1)
    assert any(parameter.requires_grad for parameter in model.classifier.parameters())
    assert not any(parameter.requires_grad for parameter in model.bert.embeddings.parameters())
    assert model.bert.encoder.layer[-1].output.dense.weight.requires_grad
    assert not model.bert.encoder.layer[0].output.dense.weight.requires_grad
    assert LABELS[1] == "B-DISEASE"


def test_resume_notes_use_metric_rounding(tmp_path: Path):
    payload = {
        "project": "clinical-nlp-extraction",
        "success_criteria": [criterion("Example", 0.8, ">=", 0.8)],
        "metrics": {
            "ncbi_disease": {
                "transformer": {"f1": 0.8614, "precision": 0.8, "recall": 0.9},
                "crf": {"f1": 0.7424, "precision": 0.7, "recall": 0.7},
            },
            "bioscope": {"held_out": {"f1": 0.8346}},
            "api": {"valid_percent": 100.0},
            "error_analysis": {"n_analyzed": 50},
        },
    }
    path = tmp_path / "RESUME_NOTES.md"
    write_resume_notes(path, payload)
    text = path.read_text(encoding="utf-8")
    assert "0.861" in text
    assert "0.742" in text
    assert "0.835" in text
    assert "100%" in text
    assert "50" in text
    figure_dir = tmp_path / "plots"
    names = write_plots(
        figure_dir,
        payload,
        {"boundary": 10, "missed": 20, "false_positive": 5},
    )
    assert (figure_dir / names[0]).exists()
