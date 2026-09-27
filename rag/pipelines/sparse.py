"""BM25, TF-IDF, and a fixed-budget packer. No query is special-cased."""
from __future__ import annotations

import math
import re
from typing import Dict, List, Sequence, Tuple

TOKEN = re.compile(r"[a-z0-9]+")
BUDGET_WORDS = 90

STOP = {
    "the", "a", "an", "of", "in", "to", "and", "is", "are", "was", "were", "for", "on",
    "that", "this", "what", "which", "who", "whom", "whose", "when", "where", "why",
    "how", "did", "does", "do", "with", "from", "by", "or", "as", "at", "be", "been",
    "it", "its", "their", "his", "her", "they", "them", "there",
}


def tokenize(text: str) -> List[str]:
    return TOKEN.findall(text.lower())


class SparseIndex:
    def __init__(self, chunks: Sequence[dict]):
        self.chunks = [dict(chunk) for chunk in chunks]
        self.docs = [tokenize(chunk.get("text") or "") for chunk in self.chunks]
        self.avgdl = sum(len(doc) for doc in self.docs) / max(len(self.docs), 1)
        df: Dict[str, int] = {}
        for doc in self.docs:
            for token in set(doc):
                df[token] = df.get(token, 0) + 1
        count = len(self.docs)
        self.idf = {
            token: math.log(1.0 + (count - freq + 0.5) / (freq + 0.5))
            for token, freq in df.items()
        }
        self.k1 = 1.5
        self.b = 0.75

    def ranked(self, query: str) -> List[Tuple[dict, float]]:
        scores = [self._bm25(query, index) for index in range(len(self.docs))]
        order = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
        return [(self.chunks[index], scores[index]) for index in order]

    def top(self, query: str, k: int) -> List[dict]:
        picked = self.ranked(query)[: max(int(k), 1)]
        return [_hit(chunk, score, rank) for rank, (chunk, score) in enumerate(picked, start=1)]

    def budget(self, query: str, k: int) -> List[dict]:
        """Pack a fixed word budget from the BM25 pool of size k.

        k=1 returns the top paragraph. A larger k still has the same word
        budget, so a long top paragraph can be left out in favor of short ones.
        """
        pool = self.ranked(query)[: max(int(k), 1)]
        if not pool:
            return []
        if int(k) <= 1:
            chunk, score = pool[0]
            return [_hit(chunk, score, 1)]
        packed = []
        used = 0
        shortest_first = sorted(pool, key=lambda item: (len(tokenize(item[0].get("text") or "")), item[0]["id"]))
        for chunk, score in shortest_first:
            words = max(len(tokenize(chunk.get("text") or "")), 1)
            if packed and used + words > BUDGET_WORDS:
                continue
            packed.append(_hit(chunk, score, len(packed) + 1))
            used += words
        return packed or [_hit(pool[0][0], pool[0][1], 1)]

    def tfidf(self, query: str, chunk_ids: Sequence[str]) -> Dict[str, float]:
        wanted = set(chunk_ids)
        query_weights = self._weights(tokenize(query))
        query_norm = math.sqrt(sum(value * value for value in query_weights.values())) or 1.0
        scores = {}
        for index, chunk in enumerate(self.chunks):
            if chunk["id"] not in wanted:
                continue
            weights = self._weights(self.docs[index])
            doc_norm = math.sqrt(sum(value * value for value in weights.values())) or 1.0
            dot = sum(query_weights.get(token, 0.0) * weight for token, weight in weights.items())
            scores[chunk["id"]] = dot / (query_norm * doc_norm)
        return scores

    def keyword_query(self, question: str, keep: int = 4) -> str:
        seen = []
        for token in tokenize(question):
            if token in STOP or token in seen:
                continue
            seen.append(token)
        ranked = sorted(seen, key=lambda token: self.idf.get(token, 0.0), reverse=True)
        return " ".join(ranked[:keep])

    def _bm25(self, query: str, index: int) -> float:
        doc = self.docs[index]
        if not doc:
            return 0.0
        counts: Dict[str, int] = {}
        for token in doc:
            counts[token] = counts.get(token, 0) + 1
        score = 0.0
        for token in tokenize(query):
            freq = counts.get(token)
            if not freq:
                continue
            denom = freq + self.k1 * (1.0 - self.b + self.b * len(doc) / self.avgdl)
            score += self.idf.get(token, 0.0) * freq * (self.k1 + 1.0) / denom
        return score

    def _weights(self, tokens: Sequence[str]) -> Dict[str, float]:
        counts: Dict[str, int] = {}
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1
        return {
            token: (1.0 + math.log(count)) * self.idf.get(token, 0.0)
            for token, count in counts.items()
        }


def _hit(chunk: dict, score: float, rank: int) -> dict:
    return {
        "id": chunk["id"],
        "text": chunk.get("text") or "",
        "score": float(score),
        "rank": rank,
    }
