"""Hand-built token features for the dictionary CRF."""

from __future__ import annotations

from clinical_nlp.lexicon import DiseaseLexicon


def word_shape(word: str) -> str:
    pieces: list[str] = []
    for char in word[:12]:
        if char.isupper():
            pieces.append("X")
        elif char.islower():
            pieces.append("x")
        elif char.isdigit():
            pieces.append("d")
        else:
            pieces.append(char)
    return "".join(pieces)


def sentence_features(tokens: list[str], lexicon: DiseaseLexicon) -> list[dict[str, str]]:
    gazetteer = lexicon.tag_tokens(tokens)
    rows: list[dict[str, str]] = []
    for index, token in enumerate(tokens):
        lowered = token.lower()
        features = {
            "bias": "1",
            "word.lower": lowered,
            "word[-3:]": lowered[-3:],
            "word[-2:]": lowered[-2:],
            "word[:3]": lowered[:3],
            "word[:2]": lowered[:2],
            "word.istitle": str(token.istitle()),
            "word.isupper": str(token.isupper()),
            "word.isdigit": str(token.isdigit()),
            "word.hyphen": str("-" in token),
            "word.shape": word_shape(token),
            "gaz": gazetteer[index],
        }
        if index == 0:
            features["BOS"] = "1"
        else:
            previous = tokens[index - 1]
            features["-1:word.lower"] = previous.lower()
            features["-1:shape"] = word_shape(previous)
            features["-1:gaz"] = gazetteer[index - 1]
        if index == len(tokens) - 1:
            features["EOS"] = "1"
        else:
            nxt = tokens[index + 1]
            features["+1:word.lower"] = nxt.lower()
            features["+1:shape"] = word_shape(nxt)
            features["+1:gaz"] = gazetteer[index + 1]
        rows.append(features)
    return rows
