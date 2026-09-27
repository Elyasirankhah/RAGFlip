"""Turn a Trace into a DiagnosisReport. This is the v0.2 product entry point."""
from __future__ import annotations

from typing import Optional

from rag.analyzer import Embedder, Judge, TraceAnalyzer
from rag.report import ClaimDiagnosis, DiagnosisReport, pipeline_from_claims
from rag.trace import Trace


def diagnose(
    trace: Trace,
    analyzer: Optional[TraceAnalyzer] = None,
    embedder: Optional[Embedder] = None,
    judge: Optional[Judge] = None,
) -> DiagnosisReport:
    if analyzer is None:
        from rag.embeddings import EmbeddingGenerator
        from rag.judge import SupportJudge

        analyzer = TraceAnalyzer(
            embedder=embedder or EmbeddingGenerator(),
            judge=judge or SupportJudge(),
        )
    raw = analyzer.analyze(
        question=trace.question,
        answer=trace.answer,
        retrieved_chunks=[chunk.to_dict() for chunk in trace.retrieved_chunks],
        corpus_chunks=[chunk.to_dict() for chunk in trace.corpus_chunks] or None,
        candidates=[chunk.to_dict() for chunk in trace.candidates] or None,
        original_query=trace.metadata.original_query,
        rewritten_query=trace.metadata.rewritten_query,
    )
    claims = [
        ClaimDiagnosis(
            claim=item.sentence,
            label=item.fault,
            confidence=item.confidence,
            best_retrieved_chunk=item.best_retrieved_chunk,
            best_corpus_chunk=item.best_corpus_chunk,
            evidence=item.evidence,
            reason=item.likely_reason or item.reason,
            component=item.component,
        )
        for item in raw.sentences
    ]
    return DiagnosisReport(
        question=trace.question,
        answer=trace.answer,
        claims=claims,
        pipeline=pipeline_from_claims(claims),
        notes=list(raw.notes),
    )
