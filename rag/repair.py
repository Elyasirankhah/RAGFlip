"""Replay a failing trace with alternative retrieval settings."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from rag.analyzer import TraceAnalyzer, _normalize_chunk, _rank
from rag.similarity import cosine_similarity


def _supported_pct(summary: Dict[str, int]) -> float:
    total = summary.get("total") or 0
    if total == 0:
        return 0.0
    return round(100.0 * summary.get("supported", 0) / total, 1)


def _lexical_score(query: str, text: str) -> float:
    q = {w.strip(".,!?").lower() for w in query.split() if len(w) > 2}
    if not q:
        return 0.0
    t = {w.strip(".,!?").lower() for w in text.split() if len(w) > 2}
    return len(q & t) / len(q)


def _take_top(chunks: List[Dict[str, str]], scores: List[float], k: int) -> List[Dict[str, str]]:
    ranked = sorted(zip(scores, chunks), key=lambda item: item[0], reverse=True)
    return [chunk for _, chunk in ranked[:k]]


def run_repair_experiments(
    analyzer: TraceAnalyzer,
    question: str,
    answer: str,
    retrieved_chunks: Sequence[Dict[str, Any]],
    corpus_chunks: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Diagnose the current trace, then replay retrieval on the provided corpus
    with larger k and a lexical+dense hybrid. Reports before vs after
    % of sentences labeled supported.
    """
    baseline = analyzer.analyze(question, answer, retrieved_chunks, corpus_chunks)
    experiments: List[Dict[str, Any]] = [
        {
            "name": "baseline",
            "supported_pct": _supported_pct(baseline.summary),
            "summary": baseline.summary,
            "root_cause": baseline.root_cause,
        }
    ]

    if not corpus_chunks:
        return {
            "baseline": baseline.to_dict(),
            "experiments": experiments,
            "best": experiments[0],
            "notes": ["corpus_chunks required to replay retrieval and prove a fix."],
        }

    corpus = [_normalize_chunk(chunk, i) for i, chunk in enumerate(corpus_chunks)]
    query_emb = analyzer.embedder.embed_text(question or "")
    corpus_embs = analyzer.embedder.embed_batch([c["text"] for c in corpus])
    dense_scores = [cosine_similarity(query_emb, emb) for emb in corpus_embs]
    lexical_scores = [_lexical_score(question, c["text"]) for c in corpus]
    hybrid_scores = [0.5 * d + 0.5 * lx for d, lx in zip(dense_scores, lexical_scores)]

    k = max(len(retrieved_chunks), 1)
    trials = [
        ("increase_k_20", dense_scores, 20),
        ("increase_k_30", dense_scores, 30),
        ("hybrid_bm25_dense_k30", hybrid_scores, 30),
        (f"increase_k_{max(k * 3, k + 5)}", dense_scores, max(k * 3, k + 5)),
    ]

    seen = {"baseline"}
    for name, scores, new_k in trials:
        if name in seen:
            continue
        seen.add(name)
        simulated = _take_top(corpus, scores, min(new_k, len(corpus)))
        report = analyzer.analyze(question, answer, simulated, corpus)
        experiments.append(
            {
                "name": name,
                "supported_pct": _supported_pct(report.summary),
                "summary": report.summary,
                "root_cause": report.root_cause,
                "k": min(new_k, len(corpus)),
            }
        )

    best = max(experiments, key=lambda item: item["supported_pct"])
    return {
        "baseline": baseline.to_dict(),
        "experiments": experiments,
        "best": best,
        "delta_supported_pct": round(best["supported_pct"] - experiments[0]["supported_pct"], 1),
        "notes": [
            "Repairs are replayed on the corpus you provided, not by mutating your production retriever."
        ],
    }
