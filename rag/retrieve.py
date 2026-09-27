"""Bounded retrieval replay. This does not return the whole corpus."""
from __future__ import annotations

from typing import Any, Dict, List, Sequence

from rag.similarity import cosine_similarity


def dense_retrieve(embedder: Any, question: str, corpus: Sequence[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
    if not corpus or k <= 0:
        return []
    query = embedder.embed_text(question or "")
    embeddings = embedder.embed_batch([str(chunk.get("text") or "") for chunk in corpus])
    scored = [
        (cosine_similarity(query, embedding), chunk)
        for chunk, embedding in zip(corpus, embeddings)
    ]
    scored.sort(key=lambda item: item[0], reverse=True)
    return [chunk for _, chunk in scored[:k]]
