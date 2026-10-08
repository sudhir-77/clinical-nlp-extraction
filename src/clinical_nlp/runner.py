"""End-to-end training and evaluation. The official test split is scored once."""

from __future__ import annotations

import csv
import gc
import json
import random
import time
from pathlib import Path

import torch
from datasets import Dataset

from clinical_nlp.baseline import predict_crf, train_crf
from clinical_nlp.bioscope import count_documents, parse_bioscope_abstracts
from clinical_nlp.download import ensure_bioscope_abstracts, ensure_ncbi
from clinical_nlp.errors import analyze
from clinical_nlp.hardware import hardware_info
from clinical_nlp.lexicon import DiseaseLexicon
from clinical_nlp.metrics import binary_scores, entity_scores
from clinical_nlp.ncbi import (
    assert_official_counts,
    parse_ncbi_file,
    sentences_from_documents,
)
from clinical_nlp.negation import is_dev_document, negation_scopes, overlaps
from clinical_nlp.ner import (
    MODEL_NAME,
    TransformerTagger,
    build_quick_model,
    build_quick_tokenizer,
    load_pretrained,
    limit_threads,
    predict_sentences,
    train_transformer,
)
from clinical_nlp.pipeline import ExtractionPipeline
from clinical_nlp.plots import write_plots
from clinical_nlp.report import criterion, update_readme, write_json, write_resume_notes
from clinical_nlp.schema import ExtractResponse

PROJECT = "clinical-nlp-extraction"
SEED = 0


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_split_counts(sentences: dict[str, list[dict]]) -> dict[str, dict[str, int]]:
    return {
        "ncbi_disease_abstracts": {"train": 593, "val": 100, "test": 100},
        "ncbi_disease_sentences": {
            "train": len(sentences["train"]),
            "val": len(sentences["dev"]),
            "test": len(sentences["test"]),
        },
    }


def _as_dataset(rows: list[dict]) -> Dataset:
    return Dataset.from_list(
        [
            {"sentence_id": row["sentence_id"], "tokens": row["tokens"], "tags": row["tags"]}
            for row in rows
        ]
    )


def _negation_rows(bioscope_sentences, lexicon: DiseaseLexicon, held_out_only: bool) -> list[dict]:
    rows: list[dict] = []
    for sentence in bioscope_sentences:
        if held_out_only and is_dev_document(sentence.doc_id):
            continue
        if not held_out_only and not is_dev_document(sentence.doc_id):
            continue
        mentions = lexicon.find(sentence.text)
        if not mentions:
            continue
        predicted = negation_scopes(sentence.text)
        for start, end, surface in mentions:
            gold = int(
                any(overlaps(start, end, left, right) for left, right in sentence.negation_spans)
            )
            pred = int(any(overlaps(start, end, left, right) for left, right in predicted))
            rows.append(
                {
                    "sentence_id": sentence.sentence_id,
                    "mention": surface,
                    "gold_negated": gold,
                    "pred_negated": pred,
                }
            )
    return rows


def _api_check(pipeline: ExtractionPipeline, notes: list[str]) -> dict[str, int]:
    from fastapi.testclient import TestClient

    from clinical_nlp.api import create_app

    valid = 0
    with TestClient(create_app(pipeline)) as client:
        for note in notes:
            response = client.post("/extract", json={"text": note})
            if response.status_code != 200:
                continue
            ExtractResponse.model_validate(response.json())
            valid += 1
    return {"n_notes": len(notes), "n_valid": valid}


def run_pipeline(
    root: Path,
    *,
    quick: bool = False,
    device_name: str = "cpu",
    results_dir: Path | None = None,
) -> dict:
    """Train on NCBI train, select on dev, score test once, then score negation."""
    limit_threads()
    started = time.perf_counter()
    results = results_dir or (root / "results")
    results.mkdir(parents=True, exist_ok=True)
    data_dir = root / "data"
    print("downloading corpora", flush=True)
    ncbi_paths = ensure_ncbi(data_dir)
    abstracts = ensure_bioscope_abstracts(data_dir)
    documents = {split: parse_ncbi_file(path) for split, path in ncbi_paths.items()}
    assert_official_counts(documents)
    sentences = {
        split: sentences_from_documents(documents[split], split)
        for split in ("train", "dev", "test")
    }
    processed = data_dir / "processed"
    for split, rows in sentences.items():
        _write_jsonl(processed / f"{split}.jsonl", rows)
    train_view = _as_dataset(sentences["train"])
    if len(train_view) != len(sentences["train"]):
        raise RuntimeError("Hugging Face dataset length does not match the train split")
    del train_view
    del documents
    gc.collect()

    train_rows = sentences["train"][:48] if quick else sentences["train"]
    dev_rows = sentences["dev"][:16] if quick else sentences["dev"]
    test_rows = sentences["test"][:16] if quick else sentences["test"]
    lexicon = DiseaseLexicon.from_sentences(train_rows)
    print(f"lexicon phrases {len(lexicon.phrases)}", flush=True)

    print("training dictionary CRF", flush=True)
    crf, crf_info = train_crf(train_rows, dev_rows, lexicon, quick=quick)

    if device_name == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    random.seed(SEED)
    torch.manual_seed(SEED)
    cache = data_dir / "hf"
    if quick:
        tokenizer = build_quick_tokenizer(train_rows + dev_rows, results / "quick_vocab.txt")
        model = build_quick_model(tokenizer)
        epochs, batch_size, max_length, n_unfreeze = 1, 8, 48, None
        min_epochs = 1
    else:
        print(f"loading {MODEL_NAME}", flush=True)
        gc.collect()
        tokenizer, model = load_pretrained(cache, n_unfreeze=12)
        # Embeddings stay frozen inside set_trainable. Every encoder layer is updated.
        epochs, batch_size, max_length, n_unfreeze = 4, 8, 128, 12
        min_epochs = 2
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print(f"trainable parameters {trainable}", flush=True)
    history = train_transformer(
        train_rows,
        dev_rows,
        tokenizer,
        model,
        device=device,
        seed=SEED,
        epochs=epochs,
        batch_size=batch_size,
        max_length=max_length,
        learning_rate=5e-5,
        n_unfreeze=n_unfreeze,
        checkpoint=results / "best_ner.pt",
        min_epochs=min_epochs,
        patience=2,
    )
    if not quick:
        model_dir = root / "models" / "ner"
        model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)

    print("scoring official test split once", flush=True)
    transformer_tags = predict_sentences(
        model, tokenizer, test_rows, max_length=max_length, device=device, batch_size=batch_size
    )
    crf_tags = predict_crf(crf, test_rows, lexicon)
    gold = [row["tags"] for row in test_rows]
    transformer_scores = entity_scores(gold, transformer_tags)
    crf_scores = entity_scores(gold, crf_tags)
    prediction_rows = []
    for row, pred, crf_row in zip(test_rows, transformer_tags, crf_tags):
        prediction_rows.append(
            {
                "sentence_id": row["sentence_id"],
                "tokens": row["tokens"],
                "gold_tags": row["tags"],
                "pred_tags": pred,
                "crf_tags": crf_row,
            }
        )
    _write_jsonl(results / "test_predictions.jsonl", prediction_rows)

    print("scoring negation on held-out BioScope abstracts", flush=True)
    bioscope_sentences = parse_bioscope_abstracts(abstracts)
    n_docs = count_documents(abstracts)
    if n_docs != 1273 or len(bioscope_sentences) != 11871:
        raise RuntimeError(
            f"Unexpected BioScope abstracts size: {n_docs} documents, "
            f"{len(bioscope_sentences)} sentences"
        )
    # Full-data lexicon for negation, so quick mode still measures the real rules.
    negation_lexicon = DiseaseLexicon.from_sentences(sentences["train"])
    held_out = _negation_rows(bioscope_sentences, negation_lexicon, held_out_only=True)
    dev_neg = _negation_rows(bioscope_sentences, negation_lexicon, held_out_only=False)
    held_scores = binary_scores(
        [row["gold_negated"] for row in held_out],
        [row["pred_negated"] for row in held_out],
    )
    dev_scores = binary_scores(
        [row["gold_negated"] for row in dev_neg],
        [row["pred_negated"] for row in dev_neg],
    )
    with (results / "negation_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sentence_id", "mention", "gold_negated", "pred_negated"],
        )
        writer.writeheader()
        writer.writerows(held_out)

    error_report = analyze(prediction_rows, limit=50)
    write_json(results / "error_analysis.json", error_report)

    notes = [row["text"] for row in test_rows if row["text"].strip()]
    if not quick:
        if len(notes) < 200:
            raise RuntimeError(f"Need 200 test notes, found {len(notes)}")
        notes = notes[:200]
    pipeline = ExtractionPipeline(
        TransformerTagger(model, tokenizer, max_length=max_length, device=device)
    )
    api_stats = _api_check(pipeline, notes)
    checkpoint = results / "best_ner.pt"
    if checkpoint.exists():
        checkpoint.unlink()
    write_json(results / "api_check.json", api_stats)

    del model, tokenizer, pipeline, crf
    gc.collect()

    gap = transformer_scores["f1"] - crf_scores["f1"]
    valid_fraction = api_stats["n_valid"] / api_stats["n_notes"] if api_stats["n_notes"] else 0.0
    api_passed = api_stats["n_notes"] >= 200 and api_stats["n_valid"] == api_stats["n_notes"]
    criteria = [
        criterion(
            "Entity-level exact-span F1 on the NCBI Disease official test split",
            transformer_scores["f1"],
            ">=",
            0.80,
        ),
        criterion("Transformer vs. CRF baseline", gap, ">=", 0.05),
        criterion(
            "Negation detection F1 on BioScope disease sentences",
            held_scores["f1"],
            ">=",
            0.80,
        ),
        criterion("API schema validity", valid_fraction, ">=", 1.0),
    ]
    criteria[-1]["passed"] = api_passed
    payload = {
        "project": PROJECT,
        "success_criteria": criteria,
        "metrics": {
            "ncbi_disease": {
                "transformer": transformer_scores,
                "crf": crf_scores,
                "f1_gap": gap,
                "dev_f1": history.get("best_dev_f1", 0.0),
                "training": history,
            },
            "bioscope": {
                "held_out": {**held_scores, "n_mentions": len(held_out)},
                "dev": {**dev_scores, "n_mentions": len(dev_neg)},
                "documents": n_docs,
                "sentences": len(bioscope_sentences),
            },
            "api": {
                "n_notes": api_stats["n_notes"],
                "n_valid": api_stats["n_valid"],
                "valid_fraction": valid_fraction,
                "valid_percent": 100.0 * valid_fraction,
            },
            "error_analysis": {
                "n_errors": error_report["n_errors"],
                "n_analyzed": error_report["n_analyzed"],
                **error_report["counts"],
            },
            "lexicon_phrases": len(negation_lexicon.phrases),
            "crf": crf_info,
            "model_name": "quick-tiny-bert" if quick else MODEL_NAME,
            "quick": quick,
        },
        "baseline": {
            "model": "dictionary_crf",
            "entity_f1": crf_scores["f1"],
            "entity_precision": crf_scores["precision"],
            "entity_recall": crf_scores["recall"],
            "c1": crf_info["c1"],
            "c2": crf_info["c2"],
        },
        "splits": {
            **_read_split_counts(sentences),
            "bioscope_abstracts": {"train": 0, "val": 0, "test": n_docs},
        },
        "seeds": [SEED],
        "hardware": hardware_info(0.0, device_name if device.type == "cuda" else "cpu"),
        "runtime_seconds": round(time.perf_counter() - started, 3),
    }
    plot_names = write_plots(results, payload, error_report["counts"])
    payload["metrics"]["plots"] = plot_names
    write_json(results / "metrics.json", payload)
    if not quick and results_dir is None:
        update_readme(root / "README.md", payload)
        write_resume_notes(root / "RESUME_NOTES.md", payload)
    print("pipeline finished", flush=True)
    return payload
