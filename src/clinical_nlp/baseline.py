"""Dictionary gazetteer plus a linear-chain CRF. Hyperparameters use dev only."""

from __future__ import annotations

import sklearn_crfsuite

from clinical_nlp.features import sentence_features
from clinical_nlp.lexicon import DiseaseLexicon
from clinical_nlp.metrics import entity_scores, repair_bio

GRID = ((0.05, 0.05), (0.1, 0.1), (0.5, 0.1), (0.1, 0.5))


def _matrices(
    sentences: list[dict], lexicon: DiseaseLexicon
) -> tuple[list[list[dict[str, str]]], list[list[str]]]:
    features = [sentence_features(row["tokens"], lexicon) for row in sentences]
    labels = [row["tags"] for row in sentences]
    return features, labels


def train_crf(
    train: list[dict],
    dev: list[dict],
    lexicon: DiseaseLexicon,
    *,
    quick: bool = False,
    max_iterations: int = 80,
) -> tuple[sklearn_crfsuite.CRF, dict[str, float]]:
    train_x, train_y = _matrices(train, lexicon)
    dev_x, dev_y = _matrices(dev, lexicon)
    grid = ((0.1, 0.1),) if quick else GRID
    iterations = 25 if quick else max_iterations
    best_model: sklearn_crfsuite.CRF | None = None
    best_f1 = -1.0
    best_c1 = grid[0][0]
    best_c2 = grid[0][1]
    for c1, c2 in grid:
        model = sklearn_crfsuite.CRF(
            algorithm="lbfgs",
            c1=c1,
            c2=c2,
            max_iterations=iterations,
            all_possible_transitions=True,
        )
        model.fit(train_x, train_y)
        predicted = [repair_bio(tags) for tags in model.predict(dev_x)]
        score = entity_scores(dev_y, predicted)["f1"] if dev_y else 0.0
        if score > best_f1:
            best_f1 = score
            best_model = model
            best_c1, best_c2 = c1, c2
    assert best_model is not None
    return best_model, {"c1": best_c1, "c2": best_c2, "dev_f1": best_f1}


def predict_crf(
    model: sklearn_crfsuite.CRF,
    sentences: list[dict],
    lexicon: DiseaseLexicon,
) -> list[list[str]]:
    features, _labels = _matrices(sentences, lexicon)
    return [repair_bio(tags) for tags in model.predict(features)]
