from clinical_nlp.metrics import binary_scores, entity_scores, repair_bio, spans_from_tags


def test_entity_f1_matches_hand_count():
    gold = [["B-DISEASE", "I-DISEASE", "O", "B-DISEASE"]]
    pred = [["B-DISEASE", "I-DISEASE", "O", "O"]]
    scores = entity_scores(gold, pred)
    assert scores["precision"] == 1.0
    assert scores["recall"] == 0.5
    assert abs(scores["f1"] - (2 / 3)) < 1e-6

    perfect = entity_scores(gold, gold)
    assert perfect["f1"] == 1.0
    none = entity_scores(
        [["B-DISEASE", "I-DISEASE"]],
        [["B-DISEASE", "O"]],
    )
    assert none["f1"] == 0.0


def test_binary_f1_hand_count():
    scores = binary_scores([1, 1, 0, 0], [1, 0, 1, 0])
    # tp=1, fp=1, fn=1 -> P=0.5, R=0.5, F=0.5
    assert scores["precision"] == 0.5
    assert scores["recall"] == 0.5
    assert scores["f1"] == 0.5
    assert scores["support"] == 2


def test_repair_and_spans():
    assert repair_bio(["O", "I-DISEASE", "I-DISEASE"]) == [
        "O",
        "B-DISEASE",
        "I-DISEASE",
    ]
    assert spans_from_tags(["B-DISEASE", "I-DISEASE", "O", "B-DISEASE"]) == [(0, 2), (3, 4)]
