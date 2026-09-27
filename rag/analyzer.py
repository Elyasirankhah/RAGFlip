"""
Trace analyzer: find faults in an existing RAG run.

This is the product brain. It does not run retrieval or generation itself.
Callers pass a question, the model's answer, the chunks that were retrieved,
and optionally the broader corpus so retrieval-miss can be separated from
hallucination.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence

from rag.fixes import fixes_for
from rag.judge import JudgeResult, SupportJudge
from rag.similarity import cosine_similarity
from utils.text_utils import split_into_sentences

MAX_SENTENCES = 40
MAX_RETRIEVED = 50
MAX_CORPUS = 200
MAX_CHUNK_CHARS = 4000
TOP_K_EVIDENCE = 3
# A single retrieved chunk must contain every content word before it can close the case.
SINGLE_CHUNK_COVERAGE = 1.0

SUPPORTED = "supported"
RETRIEVAL_MISS = "retrieval_miss"
HALLUCINATION = "hallucination"
CHUNKING_MISS = "chunking_miss"
K_TOO_SMALL = "k_too_small"
EMBEDDING_RETRIEVAL_FAILURE = "embedding_retrieval_failure"
QUERY_REWRITE_FAILURE = "query_rewrite_failure"
UNSUPPORTED = "unsupported"
UNKNOWN = "unknown"


class Embedder(Protocol):
    def embed_text(self, text: str) -> List[float]: ...
    def embed_batch(self, texts: List[str]) -> List[List[float]]: ...


class Judge(Protocol):
    def judge(self, claim: str, evidence: str, question: str = "") -> JudgeResult: ...


@dataclass
class RankedChunk:
    chunk_id: str
    text: str
    similarity: float


@dataclass
class SentenceFault:
    sentence: str
    fault: str
    component: str = ""
    supporting_chunk_ids: List[str] = field(default_factory=list)
    evidence_rank: Optional[int] = None
    retrieved_k: int = 0
    retrieved_similarity: float = 0.0
    corpus_similarity: Optional[float] = None
    likely_reason: str = ""
    suggested_fixes: List[str] = field(default_factory=list)
    reason: str = ""
    confidence: float = 0.0
    best_retrieved_chunk: Optional[str] = None
    best_corpus_chunk: Optional[str] = None
    evidence: str = ""


@dataclass
class TraceReport:
    question: str
    answer: str
    sentences: List[SentenceFault]
    summary: Dict[str, int]
    notes: List[str]
    root_cause: Optional[str] = None
    suggested_fixes: List[str] = field(default_factory=list)
    k: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "sentences": [asdict(item) for item in self.sentences],
            "summary": self.summary,
            "notes": self.notes,
            "root_cause": self.root_cause,
            "suggested_fixes": self.suggested_fixes,
            "k": self.k,
        }


def _normalize_chunk(raw: Dict[str, Any], index: int) -> Dict[str, Any]:
    text = str(raw.get("text") or "")[:MAX_CHUNK_CHARS]
    chunk_id = str(raw.get("id") or raw.get("chunk_id") or f"chunk_{index}")
    chunk = {"chunk_id": chunk_id, "text": text}
    if raw.get("rank") is not None:
        chunk["rank"] = int(raw["rank"])
    if raw.get("score") is not None:
        chunk["score"] = float(raw["score"])
    return chunk


def _join(chunks: Sequence[RankedChunk]) -> str:
    parts = [item.text.strip() for item in chunks if item.text.strip()]
    return "\n\n".join(parts)


def _rank(claim_emb: List[float], chunks: List[Dict[str, str]], embeddings: List[List[float]]) -> List[RankedChunk]:
    ranked = []
    for chunk, emb in zip(chunks, embeddings):
        ranked.append(
            RankedChunk(
                chunk_id=chunk["chunk_id"],
                text=chunk["text"],
                similarity=cosine_similarity(claim_emb, emb),
            )
        )
    ranked.sort(key=lambda item: item.similarity, reverse=True)
    return ranked


class TraceAnalyzer:
    """Classify each answer sentence as supported or a RAG fault."""

    def __init__(self, embedder: Embedder, judge: Judge):
        self.embedder = embedder
        self.judge = judge

    def analyze(
        self,
        question: str,
        answer: str,
        retrieved_chunks: Sequence[Dict[str, Any]],
        corpus_chunks: Optional[Sequence[Dict[str, Any]]] = None,
        candidates: Optional[Sequence[Dict[str, Any]]] = None,
        original_query: Optional[str] = None,
        rewritten_query: Optional[str] = None,
    ) -> TraceReport:
        notes: List[str] = []
        retrieved = [_normalize_chunk(chunk, i) for i, chunk in enumerate(retrieved_chunks[:MAX_RETRIEVED])]
        evidence_pool = corpus_chunks if corpus_chunks else candidates
        corpus = None
        if evidence_pool:
            corpus = [_normalize_chunk(chunk, i) for i, chunk in enumerate(evidence_pool[:MAX_CORPUS])]
            if not corpus_chunks:
                notes.append("Using logged candidates as the evidence pool.")
        elif not retrieved:
            notes.append("No retrieved chunks and no evidence pool were provided.")
        else:
            notes.append(
                "No corpus or candidates were logged, so retrieval miss versus hallucination is unknown."
            )

        sentences = split_into_sentences(answer)[:MAX_SENTENCES]
        if not sentences:
            return TraceReport(
                question=question,
                answer=answer,
                sentences=[],
                summary=_empty_summary(),
                notes=notes or ["Answer contained no sentences."],
            )

        sentence_embs = self.embedder.embed_batch(sentences)
        retrieved_embs = self.embedder.embed_batch([c["text"] for c in retrieved]) if retrieved else []
        corpus_embs = self.embedder.embed_batch([c["text"] for c in corpus]) if corpus else []
        logged_ranks = {
            chunk["chunk_id"]: int(chunk["rank"])
            for chunk in (corpus or [])
            if chunk.get("rank") is not None
        }
        k = len(retrieved)

        results: List[SentenceFault] = []
        for sentence, sent_emb in zip(sentences, sentence_embs):
            results.append(
                self._classify_sentence(
                    question=question,
                    sentence=sentence,
                    sent_emb=sent_emb,
                    retrieved=retrieved,
                    retrieved_embs=retrieved_embs,
                    corpus=corpus,
                    corpus_embs=corpus_embs,
                    logged_ranks=logged_ranks,
                    k=k,
                    original_query=original_query,
                    rewritten_query=rewritten_query,
                )
            )

        summary = _empty_summary()
        for item in results:
            summary[item.fault] = summary.get(item.fault, 0) + 1
            summary["total"] += 1

        root_cause, suggested = _aggregate_root_cause(results)
        return TraceReport(
            question=question,
            answer=answer,
            sentences=results,
            summary=summary,
            notes=notes,
            root_cause=root_cause,
            suggested_fixes=suggested,
            k=k,
        )

    def _classify_sentence(
        self,
        question: str,
        sentence: str,
        sent_emb: List[float],
        retrieved: List[Dict[str, str]],
        retrieved_embs: List[List[float]],
        corpus: Optional[List[Dict[str, Any]]],
        corpus_embs: List[List[float]],
        logged_ranks: Dict[str, int],
        k: int,
        original_query: Optional[str] = None,
        rewritten_query: Optional[str] = None,
    ) -> SentenceFault:
        retrieved_ranked = _rank(sent_emb, retrieved, retrieved_embs) if retrieved else []
        best_retrieved = retrieved_ranked[0].similarity if retrieved_ranked else 0.0
        retrieved_evidence = retrieved_ranked[:TOP_K_EVIDENCE]

        individual_reason = "No retrieved evidence."
        for item in retrieved_evidence:
            verdict = self.judge.judge(sentence, item.text, question)
            individual_reason = verdict.reason
            fragment = len(retrieved) > 1 and not _covers_claim(sentence, item.text)
            if verdict.label == "supported" and not fragment:
                return _fault(
                    sentence,
                    SUPPORTED,
                    SUPPORTED,
                    supporting_chunk_ids=[item.chunk_id],
                    retrieved_k=k,
                    retrieved_similarity=best_retrieved,
                    reason=verdict.reason,
                    likely_reason="Retrieved evidence entails the claim.",
                    best_retrieved_chunk=item.chunk_id,
                    evidence=item.text,
                )

        concat_verdict = self.judge.judge(
            sentence,
            _join(retrieved_ranked[: max(TOP_K_EVIDENCE, 5)]),
            question,
        )
        if retrieved_ranked and concat_verdict.label == "supported":
            return _fault(
                sentence,
                CHUNKING_MISS,
                CHUNKING_MISS,
                supporting_chunk_ids=[item.chunk_id for item in retrieved_ranked[:5]],
                retrieved_k=k,
                retrieved_similarity=best_retrieved,
                reason="Individual retrieved chunks do not entail the claim, "
                "but the chunks together do. Likely a chunking split. "
                + concat_verdict.reason,
                likely_reason="Evidence is split across retrieved chunks.",
                best_retrieved_chunk=retrieved_ranked[0].chunk_id,
                evidence=_join(retrieved_ranked[:5]),
            )

        if corpus is None:
            return _fault(
                sentence,
                UNKNOWN,
                UNKNOWN,
                retrieved_k=k,
                retrieved_similarity=best_retrieved,
                reason=individual_reason,
                likely_reason="No corpus or candidates were logged, so this claim is unknown.",
                best_retrieved_chunk=retrieved_ranked[0].chunk_id if retrieved_ranked else None,
            )

        corpus_ranked = _rank(sent_emb, corpus, corpus_embs)
        retrieved_ids = {item.chunk_id for item in retrieved_ranked}
        unseen = [item for item in corpus_ranked if item.chunk_id not in retrieved_ids]
        pool = unseen or corpus_ranked
        best_corpus = pool[0].similarity if pool else 0.0
        corpus_verdict = self.judge.judge(sentence, _join(pool[:TOP_K_EVIDENCE]), question)

        if corpus_verdict.label in {"supported", "partial"}:
            gold = pool[0]
            evidence_rank = logged_ranks.get(gold.chunk_id)
            component, likely = _retrieval_component(k, evidence_rank, question, gold.text)
            component, likely = _rewrite_component(
                component, likely, original_query, rewritten_query, gold.text
            )
            fault = QUERY_REWRITE_FAILURE if component == QUERY_REWRITE_FAILURE else RETRIEVAL_MISS
            return _fault(
                sentence,
                fault,
                component,
                supporting_chunk_ids=[item.chunk_id for item in pool[:TOP_K_EVIDENCE]],
                evidence_rank=evidence_rank,
                retrieved_k=k,
                retrieved_similarity=best_retrieved,
                corpus_similarity=best_corpus,
                reason="Claim is not supported by retrieved chunks, but it is "
                "supported by other corpus text. "
                + corpus_verdict.reason,
                likely_reason=likely,
                best_retrieved_chunk=retrieved_ranked[0].chunk_id if retrieved_ranked else None,
                best_corpus_chunk=gold.chunk_id,
                evidence=gold.text,
            )

        return _fault(
            sentence,
            HALLUCINATION,
            HALLUCINATION,
            retrieved_k=k,
            retrieved_similarity=best_retrieved,
            corpus_similarity=best_corpus,
            reason="Claim is not entailed by retrieved chunks or the provided corpus. "
            + corpus_verdict.reason,
            likely_reason="Generation produced a claim that is not in the corpus.",
            best_retrieved_chunk=retrieved_ranked[0].chunk_id if retrieved_ranked else None,
            best_corpus_chunk=pool[0].chunk_id if pool else None,
            evidence=pool[0].text if pool else "",
        )


def _content_tokens(text: str) -> set:
    skip = {"the", "a", "an", "of", "in", "to", "and", "is", "are", "was", "for", "on", "that", "this"}
    return {
        word.strip(".,!?;:\"'()[]").lower()
        for word in text.split()
        if len(word.strip(".,!?;:\"'()[]")) > 2 and word.strip(".,!?;:\"'()[]").lower() not in skip
    }


def _covers_claim(claim: str, evidence: str) -> bool:
    claim_tokens = _content_tokens(claim)
    if not claim_tokens:
        return True
    covered = len(claim_tokens & _content_tokens(evidence)) / len(claim_tokens)
    return covered >= SINGLE_CHUNK_COVERAGE


def _empty_summary() -> Dict[str, int]:
    return {
        "total": 0,
        SUPPORTED: 0,
        RETRIEVAL_MISS: 0,
        HALLUCINATION: 0,
        CHUNKING_MISS: 0,
        UNSUPPORTED: 0,
        UNKNOWN: 0,
        QUERY_REWRITE_FAILURE: 0,
    }


def _fault(
    sentence: str,
    fault: str,
    component: str,
    reason: str,
    likely_reason: str,
    supporting_chunk_ids: Optional[List[str]] = None,
    evidence_rank: Optional[int] = None,
    retrieved_k: int = 0,
    retrieved_similarity: float = 0.0,
    corpus_similarity: Optional[float] = None,
    best_retrieved_chunk: Optional[str] = None,
    best_corpus_chunk: Optional[str] = None,
    evidence: str = "",
) -> SentenceFault:
    return SentenceFault(
        sentence=sentence,
        fault=fault,
        component=component,
        supporting_chunk_ids=supporting_chunk_ids or [],
        evidence_rank=evidence_rank,
        retrieved_k=retrieved_k,
        retrieved_similarity=retrieved_similarity,
        corpus_similarity=corpus_similarity,
        likely_reason=likely_reason,
        suggested_fixes=fixes_for(component),
        reason=reason,
        confidence=_confidence(fault, retrieved_similarity, corpus_similarity),
        best_retrieved_chunk=best_retrieved_chunk,
        best_corpus_chunk=best_corpus_chunk,
        evidence=evidence,
    )


def _confidence(fault: str, retrieved_similarity: float, corpus_similarity: Optional[float]) -> float:
    """Heuristic confidence. Not a calibrated probability."""
    if fault == SUPPORTED:
        return round(min(0.95, 0.72 + max(retrieved_similarity, 0) * 0.2), 2)
    if fault == RETRIEVAL_MISS and corpus_similarity is not None:
        gap = max(0.0, corpus_similarity - retrieved_similarity)
        return round(min(0.95, 0.6 + gap * 0.3), 2)
    if fault == HALLUCINATION:
        return 0.8
    if fault == CHUNKING_MISS:
        return 0.75
    return 0.5


def _rank_of(ranked: List[RankedChunk], chunk_id: str) -> Optional[int]:
    for index, item in enumerate(ranked, start=1):
        if item.chunk_id == chunk_id:
            return index
    return None


def _missing_terms(question: str, passage: str) -> List[str]:
    stop = {"the", "a", "an", "of", "in", "to", "and", "is", "are", "was", "for", "on", "that", "this"}
    query_terms = {
        w.strip(".,!?;:").lower()
        for w in question.split()
        if len(w) > 3 and w.lower() not in stop
    }
    passage_terms = {
        w.strip(".,!?;:").lower()
        for w in passage.split()
        if len(w) > 3 and w.lower() not in stop
    }
    return sorted(passage_terms - query_terms)[:8]


def _retrieval_component(
    k: int,
    evidence_rank: Optional[int],
    question: str,
    gold_text: str,
) -> tuple:
    missing = _missing_terms(question, gold_text)
    missing_bit = f" Query embedding may have missed: {', '.join(missing)}." if missing else ""
    if evidence_rank is None:
        return RETRIEVAL_MISS, "Supporting passage was not retrieved. Rank was not logged, so the subtype is unknown."
    if evidence_rank <= k:
        return (
            RETRIEVAL_MISS,
            f"Supporting passage ranked #{evidence_rank} within k={k}, but it was not returned.",
        )
    if evidence_rank <= max(20, k * 3):
        return (
            K_TOO_SMALL,
            f"Supporting passage ranked #{evidence_rank} but k={k}.{missing_bit}",
        )
    return (
        EMBEDDING_RETRIEVAL_FAILURE,
        f"Supporting passage ranked #{evidence_rank}, far below k={k}.{missing_bit}",
    )


def _query_overlap(query: str, passage: str) -> float:
    query_tokens = _content_tokens(query)
    if not query_tokens:
        return 0.0
    return len(query_tokens & _content_tokens(passage)) / len(query_tokens)


def _rewrite_component(
    component: str,
    likely: str,
    original_query: Optional[str],
    rewritten_query: Optional[str],
    gold_text: str,
) -> tuple:
    original = (original_query or "").strip()
    rewritten = (rewritten_query or "").strip()
    if not original or not rewritten or original == rewritten:
        return component, likely
    if _query_overlap(original, gold_text) > _query_overlap(rewritten, gold_text) + 0.2:
        return (
            QUERY_REWRITE_FAILURE,
            "The original query matches the supporting passage more closely than the rewritten query.",
        )
    return component, likely


def _aggregate_root_cause(sentences: List[SentenceFault]) -> tuple:
    counts: Dict[str, int] = {}
    for item in sentences:
        if item.fault == SUPPORTED:
            continue
        counts[item.component or item.fault] = counts.get(item.component or item.fault, 0) + 1
    if not counts:
        return SUPPORTED, []
    root = max(counts, key=counts.get)
    return root, fixes_for(root)


def analyze_rag_trace(
    question: str,
    answer: str,
    retrieved_chunks: Sequence[Dict[str, Any]],
    corpus_chunks: Optional[Sequence[Dict[str, Any]]] = None,
    embedder: Optional[Embedder] = None,
    judge: Optional[Judge] = None,
) -> Dict[str, Any]:
    """Library entry point for an existing RAG pipeline."""
    from rag.embeddings import EmbeddingGenerator

    analyzer = TraceAnalyzer(
        embedder=embedder or EmbeddingGenerator(),
        judge=judge or SupportJudge(),
    )
    return analyzer.analyze(
        question=question,
        answer=answer,
        retrieved_chunks=retrieved_chunks,
        corpus_chunks=corpus_chunks,
    ).to_dict()
