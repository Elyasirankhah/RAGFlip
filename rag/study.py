"""Pre-registered selective-repair study.

Not scored. Do not point this at the sealed holdout or the mixed Hotpot set.
Do not change the selection rule in rag.selective after the first study score.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

STUDY_ID = "selective-repair-v1"
STATUS = "preregistered"
QUESTION = (
    "Can selective guarded repair reduce regression risk while maintaining "
    "competitive repair success compared with unconditional RAG repair policies?"
)
FROZEN_OUT_OF_SCOPE = ("holdout", "mixed-hotpot", "judge-dev")
REGRESSION_BUDGET = 0.05
# Secondary curve only. The primary endpoint stays 5%.
SECONDARY_REGRESSION_BUDGETS = (0.0, 0.01, 0.02, 0.05)

# Ids are sealed in rag/datasets/study before any repair score.
SPLIT_SEED = 11
ENVIRONMENTS = (
    "bm25",
    "dense",
    "hybrid",
    "rerank",
    "query_rewrite",
    "chunk_window",
    "context_budget",
)

METRICS = (
    "repair_success_under_regression_budget",
    "repair_success",
    "regression_rate",
    "abstention_rate",
    "net_repairs",
    "retriever_calls",
    "latency_s",
)

BASELINES = (
    "always_increase_k",
    "always_rerun",
    "llm_choice",
    "llm_config",
    "one_shot",
    "selective_no_guard",
    "selective",
    "oracle_best_action",
)

ABLATONS = (
    "full",
    "no_guard",
    "no_abstention",
    "no_selector",
    "no_restore_query",
    "no_merge",
    "no_rerank",
)

# Internal policy signal only. Final labels use dataset_supported plus human review.
# Exact checkpoints. The selection rule does not change with the judge.
STUDY_JUDGE_MODELS = {
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen": "Qwen/Qwen2.5-7B-Instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "gemma": "google/gemma-2-9b-it",
    "deepseek": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "phi": "microsoft/Phi-3.5-mini-instruct",
}
ROBUSTNESS_JUDGES = tuple(STUDY_JUDGE_MODELS)


def assert_not_frozen(name: str) -> None:
    """The new study cannot be scored on a set that was already used."""
    if name in FROZEN_OUT_OF_SCOPE:
        raise ValueError(f"{name} is frozen and out of scope for {STUDY_ID}")


def dataset_supported(gold: str, context: str) -> bool:
    """Scientific label when a dataset gives the answer string."""
    needle = " ".join(gold.casefold().split())
    haystack = " ".join(context.casefold().split())
    if not needle:
        return False
    return needle in haystack


def repair_under_budget(
    repairs: int,
    regressions: int,
    controls: int,
    budget: float = REGRESSION_BUDGET,
) -> Optional[int]:
    """Repairs kept only when the control regression rate is within budget."""
    if controls <= 0:
        return None
    if regressions / controls > budget:
        return None
    return repairs


def summarize_outcomes(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate one policy. This does not load a benchmark."""
    failures = [row for row in rows if row.get("kind") == "failure"]
    controls = [row for row in rows if row.get("kind") == "control"]
    repaired = sum(1 for row in failures if row.get("repaired"))
    regressed = sum(1 for row in controls if row.get("regressed"))
    abstained = sum(1 for row in failures if row.get("abstained"))
    return {
        "failures": len(failures),
        "controls": len(controls),
        "repair_success": _rate(repaired, len(failures)),
        "regression_rate": _rate(regressed, len(controls)),
        "abstention_rate": _rate(abstained, len(failures)),
        "net_repairs": repaired - regressed,
        "repair_under_budget": repair_under_budget(repaired, regressed, len(controls)),
        "repair_under_budgets": {
            _budget_key(budget): repair_under_budget(repaired, regressed, len(controls), budget)
            for budget in SECONDARY_REGRESSION_BUDGETS
        },
        "retriever_calls": sum(int(row.get("calls") or 0) for row in rows),
    }


def _budget_key(budget: float) -> str:
    return f"{budget:.0%}"


def _rate(count: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return round(count / total, 3)
