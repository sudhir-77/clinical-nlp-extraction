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
        self.unigrams = {phrase[0] for phrase in phrases if len(phrase) == 1}
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

    def _variant_unigram(self, token: str) -> str | None:
        """Map a numeric strain suffix or a regular plural back to a training unigram.

        NCBI stores the base form (HIV, lymphoma). BioScope often writes HIV-1
        or lymphomas. The hyphen suffix must be all digits. A plural must use
        -s, -es, or -ies.
        """
        if "-" in token:
            head, _, tail = token.partition("-")
            if tail.isdigit() and head in self.unigrams:
                return head
        if len(token) <= 4 or not token.endswith("s"):
            return None
        stems: list[str] = []
        if token.endswith("ies"):
            stems.append(token[:-3] + "y")
        if token.endswith("es"):
            stems.append(token[:-2])
        stems.append(token[:-1])
        for stem in stems:
            if stem in self.unigrams:
                return stem
        return None

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
            if length == 0 and self._variant_unigram(words[index]):
                length = 1
            if length:
                start = spans[index][0]
                end = spans[index + length - 1][1]
                found.append((start, end, text[start:end]))
                index += length
            else:
                index += 1
        return found
