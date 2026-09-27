"""Run the smallest retriever change and check the failed claim."""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence

from rag.diagnose import diagnose
from rag.trace import Chunk, Trace

Retriever = Callable[[str, int], Sequence[Any]]


def choose_experiment(trace: Trace, report: Any) -> Optional[Dict[str, Any]]:
    """Pick one change. The planted label is not an input."""
    failed = [claim for claim in report.claims if claim.label != "supported"]
    if not failed:
        return None
    component = failed[0].component or failed[0].label
    k = trace.metadata.top_k or len(trace.retrieved_chunks) or 1
    query = trace.question
    if component == "query_rewrite_failure" and trace.metadata.original_query:
        return _experiment(
            "restore_original_query",
            trace.metadata.original_query,
            k,
            "Retrieve again with the original query.",
            from_k=k,
        )
    if component in {"retrieval_miss", "k_too_small", "embedding_retrieval_failure"} and _rerank_disagrees(trace):
        return _experiment(
            "rerank_candidates",
            query,
            k,
            "Reorder the logged candidates with the rerank score.",
            from_k=k,
        )
    if component == "k_too_small":
        return _experiment(
            "increase_k",
            query,
            min(20, max(k * 3, k + 5)),
            "Retrieve a larger candidate set.",
            from_k=k,
        )
    if component == "embedding_retrieval_failure":
        return _experiment(
            "increase_k",
            query,
            min(30, max(k * 3, 20)),
            "Retrieve more candidates with the same query.",
            from_k=k,
        )
    if component == "retrieval_miss":
        return _experiment(
            "increase_k",
            query,
            min(20, max(k * 3, k + 5)),
            "Retrieve a larger candidate set.",
            from_k=k,
        )
    if component == "chunking_miss":
        return _experiment(
            "merge_retrieved",
            query,
            k,
            "Merge the retrieved chunks into one context window.",
            runnable=True,
            from_k=k,
        )
    return _experiment(
        "no_retrieval_change",
        query,
        k,
        "No retrieval change can supply evidence that is absent from the corpus.",
        runnable=False,
        from_k=k,
    )


def run_verified_repair(
    trace: Trace,
    retriever: Retriever,
    analyzer: Any = None,
) -> Dict[str, Any]:
    """Diagnose, run one retriever call, and check each failed claim."""
    before = diagnose(trace, analyzer=analyzer)
    return execute_experiment(trace, retriever, analyzer, choose_experiment(trace, before), before)


def execute_experiment(
    trace: Trace,
    retriever: Retriever,
    analyzer: Any,
    experiment: Optional[Dict[str, Any]],
    before: Any,
) -> Dict[str, Any]:
    """Run one already chosen change and check the failed claim."""
    if experiment is None:
        return {
            "experiment": None,
            "claims": [],
            "failed_claims": 0,
            "failed_claims_improved": 0,
            "regressions": 0,
            "verified": False,
            "judge_supported": False,
            "recommendation": "UNCERTAIN",
            "note": "No failed claim to repair.",
            "before": before.to_dict(),
            "after": None,
        }
    after = None
    retrieved: List[Chunk] = []
    if experiment["runnable"]:
        if experiment.get("name") == "merge_retrieved":
            merged = "\n\n".join(chunk.text.strip() for chunk in trace.retrieved_chunks if chunk.text.strip())
            retrieved = [Chunk(id="merged", text=merged)]
        elif experiment.get("name") == "rerank_candidates":
            ranked = sorted(trace.candidates, key=lambda chunk: chunk.score or 0.0, reverse=True)
            retrieved = ranked[: max(int(experiment["k"]), 1)]
        else:
            retrieved = _as_chunks(retriever(experiment["query"], int(experiment["k"])))
        after_trace = Trace(
            question=trace.question,
            answer=trace.answer,
            retrieved_chunks=retrieved,
            corpus_chunks=list(trace.corpus_chunks),
            candidates=list(trace.candidates),
            metadata=trace.metadata,
        )
        after = diagnose(after_trace, analyzer=analyzer)
    claims = _claim_rows(before, after)
    failed = [row for row in claims if row["before"] != "supported"]
    improved = [row for row in failed if row["improved"]]
    regressions = [row for row in claims if row["regressed"]]
    verified = bool(experiment["runnable"] and failed and len(improved) == len(failed) and not regressions)
    return {
        "experiment": experiment,
        "retrieved_ids": [chunk.id for chunk in retrieved],
        "claims": claims,
        "failed_claims": len(failed),
        "failed_claims_improved": len(improved),
        "regressions": len(regressions),
        "verified": verified,
        "judge_supported": verified,
        "recommendation": repair_recommendation(experiment, verified, len(regressions), len(failed)),
        "before": before.to_dict(),
        "after": after.to_dict() if after is not None else None,
    }


def repair_recommendation(experiment: Optional[Dict[str, Any]], verified: bool, regressions: int, failed: int) -> str:
    """ACCEPT, REJECT, or UNCERTAIN. This is the product decision, not ground truth."""
    if not experiment or not experiment.get("runnable") or failed <= 0:
        return "UNCERTAIN"
    if regressions:
        return "REJECT EXPERIMENT"
    if verified:
        return "ACCEPT EXPERIMENT"
    return "REJECT EXPERIMENT"


def format_repair(result: Dict[str, Any]) -> str:
    experiment = result.get("experiment") or {}
    claims = list(result.get("claims") or [])
    failed_rows = [row for row in claims if row.get("before") != "supported"]
    supported_before = [row for row in claims if row.get("before") == "supported"]
    regressions = int(result.get("regressions", 0) or 0)
    failed = int(result.get("failed_claims", len(failed_rows)) or 0)
    verified = bool(result.get("judge_supported", result.get("verified")))
    recommendation = result.get("recommendation") or repair_recommendation(experiment, verified, regressions, failed)
    headline = failed_rows[0] if failed_rows else (claims[0] if claims else None)
    lines = ["RAG Debugger"]
    if result.get("judge"):
        lines.append(f"Judge: {result['judge']}")
    lines.append("")
    if not experiment:
        lines.append(result.get("note") or "Nothing to repair.")
        lines.append("")
        lines.append("Recommendation")
        lines.append("UNCERTAIN")
        return "\n".join(lines).rstrip() + "\n"
    lines.append("Failed claim")
    lines.append(f"\"{(headline or {}).get('claim') or '(none)'}\"")
    lines.append("")
    lines.append("Diagnosis")
    lines.append(_plain_failure(_diagnosis_label(result, headline)))
    lines.append("")
    lines.append("Experiment")
    lines.append(_experiment_line(experiment))
    if not experiment.get("runnable"):
        lines.append("Not run. No retrieval change was applied.")
    lines.append("")
    lines.append("Before")
    lines.append(_support_word((headline or {}).get("before")))
    lines.append("")
    lines.append("After")
    lines.append(_support_word((headline or {}).get("after")))
    lines.append("")
    lines.append("Regression check")
    checked = len(supported_before)
    claim_word = "claim" if checked == 1 else "claims"
    regression_word = "regression" if regressions == 1 else "regressions"
    lines.append(f"{checked} previously supported {claim_word} checked")
    lines.append(f"{regressions} {regression_word}")
    lines.append("")
    lines.append(f"judge_supported={'true' if verified else 'false'}")
    lines.append("")
    lines.append("Recommendation")
    lines.append(recommendation)
    return "\n".join(lines).rstrip() + "\n"


def retriever_from_corpus(embedder: Any, corpus: Sequence[Any]) -> Retriever:
    """A retriever that ranks a fixed corpus. Used when the caller has no live index."""
    from rag.retrieve import dense_retrieve

    pool = [chunk.to_dict() if isinstance(chunk, Chunk) else chunk for chunk in corpus]

    def retrieve(query: str, k: int) -> List[Dict[str, Any]]:
        return dense_retrieve(embedder, query, pool, k)

    return retrieve


def _rerank_disagrees(trace: Trace) -> bool:
    """True when a logged rerank score prefers a chunk the retriever did not return."""
    scored = [chunk for chunk in trace.candidates if chunk.score is not None]
    if not scored:
        return False
    best = max(scored, key=lambda chunk: chunk.score or 0.0)
    return best.id not in {chunk.id for chunk in trace.retrieved_chunks}


def _experiment(
    name: str,
    query: str,
    k: int,
    why: str,
    runnable: bool = True,
    from_k: Optional[int] = None,
) -> Dict[str, Any]:
    data = {"name": name, "query": query, "k": k, "why": why, "runnable": runnable}
    if from_k is not None:
        data["from_k"] = from_k
    return data


def _diagnosis_label(result: Dict[str, Any], headline: Optional[Dict[str, Any]]) -> str:
    before = result.get("before") or {}
    if headline and isinstance(before, dict):
        for claim in before.get("claims") or []:
            if claim.get("claim") == headline.get("claim") and claim.get("component"):
                return str(claim["component"])
    pipeline = before.get("pipeline") if isinstance(before, dict) else None
    if isinstance(pipeline, dict) and pipeline.get("primary_failure"):
        return str(pipeline["primary_failure"])
    if headline:
        return str(headline.get("before") or "")
    return ""


def _plain_failure(label: str) -> str:
    if label in {"retrieval", "retrieval_miss", "k_too_small", "embedding_retrieval_failure"}:
        return "Likely retrieval failure"
    if label in {"chunking", "chunking_miss"}:
        return "Likely chunking failure"
    if label == "query_rewrite_failure":
        return "Likely query rewrite failure"
    if label in {"generation", "hallucination"}:
        return "Likely unsupported claim"
    if label in {"supported", "none"}:
        return "Already supported"
    return "Unclear"


def _support_word(label: Optional[str]) -> str:
    if label == "supported":
        return "supported"
    if label in {None, "", "not_rerun"}:
        return "not rerun"
    return "unsupported"


def _experiment_line(experiment: Dict[str, Any]) -> str:
    name = str(experiment.get("name") or "none")
    k = experiment.get("k")
    src = experiment.get("from_k")
    if name == "increase_k" and src is not None and k is not None:
        return f"increase_k: {src} → {k}"
    if k is None:
        return name
    return f"{name} (k={k})"


def _as_chunks(raw_chunks: Sequence[Any]) -> List[Chunk]:
    chunks = []
    for index, raw in enumerate(raw_chunks or []):
        if isinstance(raw, Chunk):
            chunks.append(raw)
            continue
        if isinstance(raw, dict):
            rank = raw.get("rank")
            score = raw.get("score")
            chunks.append(
                Chunk(
                    id=str(raw.get("id") or raw.get("chunk_id") or f"chunk_{index}"),
                    text=str(raw.get("text") or raw.get("page_content") or ""),
                    rank=int(rank) if rank is not None else None,
                    score=float(score) if score is not None else None,
                )
            )
            continue
        chunks.append(Chunk(id=f"chunk_{index}", text=str(raw)))
    return chunks


def _claim_rows(before: Any, after: Optional[Any]) -> List[Dict[str, Any]]:
    after_labels = {}
    if after is not None:
        after_labels = {claim.claim: claim.label for claim in after.claims}
    rows = []
    for claim in before.claims:
        after_label = after_labels.get(claim.claim, "not_rerun")
        rows.append(
            {
                "claim": claim.claim,
                "before": claim.label,
                "after": after_label,
                "improved": claim.label != "supported" and after_label == "supported",
                "regressed": claim.label == "supported" and after_label not in {"supported", "not_rerun"},
            }
        )
    return rows
