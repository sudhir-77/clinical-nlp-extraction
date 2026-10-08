from clinical_nlp.baseline import predict_crf, train_crf
from clinical_nlp.lexicon import DiseaseLexicon


def _row(sentence_id: str, tokens: list[str], tags: list[str]) -> dict:
    return {
        "sentence_id": sentence_id,
        "tokens": tokens,
        "tags": tags,
        "text": " ".join(tokens),
    }


def test_lexicon_longest_match_and_crf_learns_dictionary_token():
    train = []
    for index in range(12):
        train.append(_row(f"t{index}", ["flu", "is", "severe"], ["B-DISEASE", "O", "O"]))
        train.append(_row(f"n{index}", ["the", "cat", "sat"], ["O", "O", "O"]))
    dev = [
        _row("d1", ["flu", "is", "mild"], ["B-DISEASE", "O", "O"]),
        _row("d2", ["the", "cat", "ran"], ["O", "O", "O"]),
    ]
    lexicon = DiseaseLexicon.from_sentences(train)
    assert lexicon.longest_at(["severe", "flu", "today"], 1) == 1
    assert lexicon.find("Today flu is severe.")[0][2].lower() == "flu"
    model, info = train_crf(train, dev, lexicon, quick=True, max_iterations=40)
    assert info["dev_f1"] == 1.0
    predicted = predict_crf(model, [_row("new", ["acute", "flu", "case"], ["O", "B-DISEASE", "O"])], lexicon)
    assert predicted[0][1] == "B-DISEASE"
