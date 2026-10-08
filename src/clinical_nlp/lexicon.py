"""Disease phrase lexicon built only from NCBI training mentions."""

from __future__ import annotations

from collections import defaultdict

from clinical_nlp.tokenize import tokenize_with_spans

STOPWORDS = {
    "a", "an", "the", "of", "and", "or", "in", "on", "to", "for", "with", "by",
    "from", "at", "as", "is", "was", "were", "be", "been", "are", "this", "that",
    "these", "those", "it", "its", "their", "his", "her", "than", "into", "over",
    "under", "between", "after", "before", "during", "within", "without", "not",
    "no",
}


def phrase_allowed(tokens: tuple[str, ...]) -> bool:
    words = [token for token in tokens if any(char.isalnum() for char in token)]
    if not words:
        return False
    if len(words) == 1 and (len(words[0]) < 3 or words[0] in STOPWORDS):
        return False
    if all(word in STOPWORDS for word in words):
        return False
    return True


class DiseaseLexicon:
    def __init__(self, phrases: set[tuple[str, ...]]) -> None:
        self.phrases = phrases
        self.by_first: dict[str, list[tuple[str, ...]]] = defaultdict(list)
        for phrase in phrases:
            self.by_first[phrase[0]].append(phrase)
        for bucket in self.by_first.values():
            bucket.sort(key=len, reverse=True)

    @classmethod
    def from_sentences(cls, sentences: list[dict]) -> "DiseaseLexicon":
        phrases: set[tuple[str, ...]] = set()
        for sentence in sentences:
            tokens = [token.lower() for token in sentence["tokens"]]
            tags = sentence["tags"]
            index = 0
            while index < len(tags):
                if tags[index] == "B-DISEASE":
                    end = index + 1
                    while end < len(tags) and tags[end] == "I-DISEASE":
                        end += 1
                    phrase = tuple(tokens[index:end])
                    if phrase_allowed(phrase):
                        phrases.add(phrase)
                    index = end
                else:
                    index += 1
        return cls(phrases)

    def longest_at(self, words: list[str], index: int) -> int:
        best = 0
        for phrase in self.by_first.get(words[index], ()):
            length = len(phrase)
            if length <= best or index + length > len(words):
                continue
            if tuple(words[index : index + length]) == phrase:
                best = length
        return best

    def tag_tokens(self, tokens: list[str]) -> list[str]:
        words = [token.lower() for token in tokens]
        tags = ["O"] * len(words)
        index = 0
        while index < len(words):
            length = self.longest_at(words, index)
            if length:
                tags[index] = "B"
                for offset in range(1, length):
                    tags[index + offset] = "I"
                index += length
            else:
                index += 1
        return tags

    def find(self, text: str) -> list[tuple[int, int, str]]:
        tokens, spans = tokenize_with_spans(text)
        words = [token.lower() for token in tokens]
        found: list[tuple[int, int, str]] = []
        index = 0
        while index < len(words):
            length = self.longest_at(words, index)
            if length:
                start = spans[index][0]
                end = spans[index + length - 1][1]
                found.append((start, end, text[start:end]))
                index += length
            else:
                index += 1
        return found
