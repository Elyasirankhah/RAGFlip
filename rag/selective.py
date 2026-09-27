"""Selective guarded repair.

Generate every legal repair, run each one, then apply a change only when it
fixes the failed claims and regresses none. Otherwise abstain. The internal
judge is the policy signal. It is not the study's ground truth.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence

from rag.diagnose import diagnose
from rag.experiment import Retriever, _experiment, _rerank_disagrees, execute_experiment
from rag.trace import Trace

# Tie-break only. Every runnable action is executed before this order is used.
# Local edits come before another retriever call. Wider k is last.
ACTION_ORDER = (
    "merge_retrieved",
    "rerank_candidates",
    "restore_original_query",
    "rerun_retriever",
    "increase_k",
    "abstain",
)
MODES = ("full", "no_guard", "no_abstention")


def candidate_actions(trace: Trace, disabled: Sequence[str] = ()) -> List[Dict[str, Any]]:
    """Legal actions for this trace. The failure label does not filter them."""
    blocked = set(disabled)
    k = trace.metadata.top_k or len(trace.retrieved_chunks) or 1
    query = trace.question
    actions: List[Dict[str, Any]] = []
    if "merge_retrieved" not in blocked and sum(1 for chunk in trace.retrieved_chunks if chunk.text.strip()) >= 2:
        actions.append(
            _experiment(
                "merge_retrieved",
                query,
                k,
                "Merge the retrieved chunks into one context window.",
                from_k=k,
            )
        )
    if "rerank_candidates" not in blocked and _rerank_disagrees(trace):
        actions.append(
            _experiment(
                "rerank_candidates",
                query,
                k,
                "Reorder the logged candidates with the rerank score.",
                from_k=k,
            )
        )
    if "restore_original_query" not in blocked and trace.metadata.original_query:
        actions.append(
            _experiment(
                "restore_original_query",
                trace.metadata.original_query,
                k,
                "Retrieve again with the original query.",
                from_k=k,
            )
        )
    if "rerun_retriever" not in blocked:
        actions.append(
            _experiment(
                "rerun_retriever",
                query,
                k,
                "Call the retriever again at the same k.",
                from_k=k,
            )
        )
    if "increase_k" not in blocked:
        actions.append(
            _experiment(
                "increase_k",
                query,
                min(20, max(k * 3, k + 5)),
                "Retrieve a larger candidate set.",
                from_k=k,
            )
        )
    actions.append(
        _experiment(
            "abstain",
            query,
            k,
            "Abstain. No candidate repaired the failure without a regression.",
            runnable=False,
            from_k=k,
        )
    )
    return actions


def choose_scored(rows: Sequence[Dict[str, Any]], mode: str = "full") -> str:
    """Pick one measured action. This rule is frozen before the study is scored."""
    if mode not in MODES:
        raise ValueError(f"Unknown selection mode: {mode}")
    ranked = [row for row in rows if row["name"] != "abstain"]
    if not ranked:
        return "abstain"

    def order(row: Dict[str, Any]) -> int:
        return ACTION_ORDER.index(row["name"])

    if mode == "full":
        safe = [row for row in ranked if row["improvement"] >= 1 and int(row["regressions"]) == 0]
        if not safe:
            return "abstain"
        return min(safe, key=lambda row: (int(row["calls"]), order(row)))["name"]

    if mode == "no_guard":
        repaired = [row for row in ranked if row["improvement"] >= 1]
        if not repaired:
            return "abstain"
        return min(repaired, key=lambda row: (int(row["regressions"]), int(row["calls"]), order(row)))["name"]

    return min(
        ranked,
        key=lambda row: (-float(row["improvement"]), int(row["regressions"]), int(row["calls"]), order(row)),
    )["name"]


def select_repair(
    trace: Trace,
    retriever: Retriever,
    analyzer: Any = None,
    mode: str = "full",
    disabled: Sequence[str] = (),
) -> Dict[str, Any]:
    """Run every candidate, then repair or abstain."""
    if mode not in MODES:
        raise ValueError(f"Unknown selection mode: {mode}")
    before = diagnose(trace, analyzer=analyzer)
    failed = [claim for claim in before.claims if claim.label != "supported"]
    actions = candidate_actions(trace, disabled)
    abstain = next(action for action in actions if action["name"] == "abstain")
    if not failed or any(claim.label == "unknown" for claim in failed):
        result = execute_experiment(trace, retriever, analyzer, abstain, before)
        result["recommendation"] = "ABSTAIN"
        result["tried"] = []
        result["retriever_calls"] = 0
        result["policy"] = "selective"
        result["mode"] = mode
        return result

    failed_n = len(failed)
    tried: List[Dict[str, Any]] = []
    results: Dict[str, Dict[str, Any]] = {}
    calls = 0
    for action in actions:
        if action["name"] == "abstain":
            continue
        started = time.perf_counter()
        result = execute_experiment(trace, retriever, analyzer, action, before)
        elapsed = time.perf_counter() - started
        action_calls = 0 if action["name"] in {"merge_retrieved", "rerank_candidates"} else 1
        calls += action_calls
        improved = int(result.get("failed_claims_improved") or 0)
        row = {
            "name": action["name"],
            "k": action.get("k"),
            "improvement": improved / failed_n,
            "regressions": int(result.get("regressions") or 0),
            "calls": action_calls,
            "latency_s": round(elapsed, 6),
            "verified": bool(result.get("verified")),
        }
        tried.append(row)
        results[action["name"]] = result

    chosen = choose_scored(tried, mode)
    if chosen == "abstain":
        result = execute_experiment(trace, retriever, analyzer, abstain, before)
        result["recommendation"] = "ABSTAIN"
    else:
        result = results[chosen]
        result["recommendation"] = "ACCEPT EXPERIMENT"
    result["tried"] = tried
    result["retriever_calls"] = calls
    result["policy"] = "selective"
    result["mode"] = mode
    return result
