"""Developer-facing diagnosis report for one RAG trace."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from rag.fixes import fixes_for


@dataclass
class ClaimDiagnosis:
    claim: str
    label: str
    confidence: float
    best_retrieved_chunk: Optional[str]
    best_corpus_chunk: Optional[str]
    evidence: str
    reason: str
    component: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PipelineDiagnosis:
    supported: int
    total: int
    retrieval_misses: int
    hallucinations: int
    chunking_misses: int
    primary_failure: str
    summary: str
    test_next: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DiagnosisReport:
    question: str
    answer: str
    claims: List[ClaimDiagnosis]
    pipeline: PipelineDiagnosis
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "claims": [claim.to_dict() for claim in self.claims],
            "pipeline": self.pipeline.to_dict(),
            "notes": self.notes,
        }

    def format(self) -> str:
        lines = [
            self.pipeline.summary,
            "",
            f"Primary suspected failure: {self.pipeline.primary_failure}",
            f"Test next: {self.pipeline.test_next}",
            "",
        ]
        marks = {
            "supported": "[ok]",
            "hallucination": "[x]",
            "retrieval_miss": "[!]",
            "chunking_miss": "[!]",
            "unknown": "[?]",
        }
        for index, claim in enumerate(self.claims, start=1):
            mark = marks.get(claim.label, "[?]")
            lines.append(f"{mark} Claim {index} — {claim.label} (confidence {claim.confidence:.2f})")
            lines.append(f"    {claim.claim}")
            if claim.best_retrieved_chunk:
                lines.append(f"    best retrieved: {claim.best_retrieved_chunk}")
            if claim.best_corpus_chunk:
                lines.append(f"    best corpus: {claim.best_corpus_chunk}")
            if claim.evidence:
                lines.append(f"    evidence: {claim.evidence[:240]}")
            lines.append(f"    reason: {claim.reason}")
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"


def pipeline_from_claims(claims: List[ClaimDiagnosis]) -> PipelineDiagnosis:
    total = len(claims)
    supported = sum(1 for claim in claims if claim.label == "supported")
    retrieval = sum(1 for claim in claims if claim.label == "retrieval_miss")
    hallucinations = sum(1 for claim in claims if claim.label == "hallucination")
    chunking = sum(1 for claim in claims if claim.label == "chunking_miss")
    counts = {
        "retrieval": retrieval,
        "chunking": chunking,
        "generation": hallucinations,
    }
    if supported == total and total:
        primary = "none"
    elif any(counts.values()):
        primary = max(counts, key=counts.get)
    else:
        primary = "unknown"

    outside = [
        claim for claim in claims
        if claim.component in {"k_too_small", "embedding_retrieval_failure"}
    ]
    summary_bits = [f"{supported}/{total} claims supported"]
    if retrieval:
        summary_bits.append(f"{retrieval} retrieval misses")
    if chunking:
        summary_bits.append(f"{chunking} chunking misses")
    if hallucinations:
        summary_bits.append(f"{hallucinations} hallucinations")
    if outside:
        summary_bits.append("Relevant evidence typically ranked outside the retrieved set")

    component = ""
    for claim in claims:
        if claim.label != "supported" and claim.component:
            component = claim.component
            break
    fixes = fixes_for(component or primary)
    test_next = fixes[0] if fixes else "Pass corpus_chunks so retrieval miss can be separated from hallucination."

    return PipelineDiagnosis(
        supported=supported,
        total=total,
        retrieval_misses=retrieval,
        hallucinations=hallucinations,
        chunking_misses=chunking,
        primary_failure=primary,
        summary=". ".join(summary_bits) + ".",
        test_next=test_next,
    )


def compare_reports(before: Dict[str, Any], after: Dict[str, Any]) -> str:
    left = before.get("pipeline") or {}
    right = after.get("pipeline") or {}

    def pct(pipe: Dict[str, Any]) -> float:
        total = pipe.get("total") or 0
        if not total:
            return 0.0
        return round(100.0 * (pipe.get("supported") or 0) / total, 1)

    before_pct = pct(left)
    after_pct = pct(right)
    delta = round(after_pct - before_pct, 1)
    sign = "+" if delta > 0 else ""
    lines = [
        f"Grounded claims: {before_pct}% -> {after_pct}% ({sign}{delta})",
        f"Retrieval misses: {left.get('retrieval_misses', 0)} -> {right.get('retrieval_misses', 0)}",
        f"Hallucinations: {left.get('hallucinations', 0)} -> {right.get('hallucinations', 0)}",
        f"Primary failure: {left.get('primary_failure', '?')} -> {right.get('primary_failure', '?')}",
    ]
    before_claims = {
        claim.get("claim"): claim.get("label")
        for claim in before.get("claims") or []
        if isinstance(claim, dict)
    }
    after_claims = {
        claim.get("claim"): claim.get("label")
        for claim in after.get("claims") or []
        if isinstance(claim, dict)
    }
    for claim, label in before_claims.items():
        after_label = after_claims.get(claim, "missing")
        if label != after_label:
            lines.append(f"Claim: {label} -> {after_label}: {claim}")
    return "\n".join(lines) + "\n"
