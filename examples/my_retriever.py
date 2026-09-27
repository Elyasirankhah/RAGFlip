"""Search the text files in examples/corpus. It does not hardcode an answer."""
from pathlib import Path

CORPUS = Path(__file__).resolve().parent / "corpus"


def retrieve(query: str, k: int):
    query_terms = _terms(query)
    scored = []
    for path in sorted(CORPUS.glob("*.txt")):
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        overlap = len(query_terms & _terms(text))
        scored.append((overlap, path.stem, text))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [{"id": name, "text": text} for _, name, text in scored[: max(k, 1)]]


def _terms(text: str) -> set:
    skip = {"the", "a", "an", "of", "in", "to", "and", "is", "are", "was", "for", "on", "that", "this", "what"}
    return {word.strip(".,!?;:\"'()[]").lower() for word in text.split() if len(word) > 2 and word.lower() not in skip}
