"""Suggested repairs for a diagnosed RAG component failure."""
from typing import Dict, List

FIXES: Dict[str, List[str]] = {
    "k_too_small": [
        "Increase candidate retrieval (for example 10 → 30) before reranking",
        "Add a cross-encoder reranker on the larger candidate set",
    ],
    "embedding_retrieval_failure": [
        "Add hybrid BM25 + dense retrieval",
        "Rewrite the query or use HyDE so missing passage terms enter the query embedding",
    ],
    "query_rewrite_failure": [
        "Keep the original user query alongside the rewritten query",
        "Constrain the rewriter so it cannot drop rare but critical terms",
    ],
    "chunking_miss": [
        "Increase chunk size or overlap so evidence stays in one window",
        "Use parent-child / small-to-big retrieval",
    ],
    "hallucination": [
        "Require citations and abstain when no chunk entails the claim",
        "Lower temperature and tighten the grounded-generation prompt",
    ],
    "retrieval_miss": [
        "Check why a top-ranked passage was dropped from the retrieved set",
        "If the passage ranks below k, raise k or add hybrid search",
    ],
    "unsupported": [
        "Pass corpus_chunks (or extra neighbors) so retrieval miss can be separated from hallucination",
    ],
    "unknown": [
        "Log candidate chunks with rank and score before separating retrieval from hallucination",
    ],
}


def fixes_for(component: str) -> List[str]:
    return list(FIXES.get(component, FIXES.get("retrieval_miss", [])))
