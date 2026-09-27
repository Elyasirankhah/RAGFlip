import pytest

from rag.analyzer import TraceAnalyzer
from rag.judge import JudgeResult
from rag.selective import choose_scored, select_repair
from rag.study import assert_not_frozen, dataset_supported, repair_under_budget, summarize_outcomes
from rag.trace import Chunk, Trace, TraceMetadata
from tests.test_analyzer import FakeEmbedder, FakeJudge


def _analyzer():
    return TraceAnalyzer(FakeEmbedder(), FakeJudge())


def test_choose_scored_prefers_a_free_safe_repair():
    rows = [
        {"name": "increase_k", "improvement": 1, "regressions": 0, "calls": 1},
        {"name": "merge_retrieved", "improvement": 1, "regressions": 0, "calls": 0},
        {"name": "rerun_retriever", "improvement": 0, "regressions": 0, "calls": 1},
    ]
    assert choose_scored(rows, "full") == "merge_retrieved"


def test_choose_scored_abstains_when_the_repair_regresses():
    rows = [
        {"name": "increase_k", "improvement": 1, "regressions": 1, "calls": 1},
        {"name": "rerun_retriever", "improvement": 0, "regressions": 0, "calls": 1},
    ]
    assert choose_scored(rows, "full") == "abstain"
    assert choose_scored(rows, "no_guard") == "increase_k"


def test_selective_runs_every_candidate_and_keeps_merge():
    trace = Trace(
        question="When was CERN founded?",
        answer="CERN was founded in 1961.",
        retrieved_chunks=[
            Chunk(id="left", text="CERN was founded"),
            Chunk(id="right", text="in 1961 near Geneva."),
        ],
        corpus_chunks=[
            Chunk(id="left", text="CERN was founded"),
            Chunk(id="right", text="in 1961 near Geneva."),
        ],
        metadata=TraceMetadata(top_k=2),
    )
    called = []

    def retrieve(query, k):
        called.append(k)
        return [{"id": "full", "text": "CERN was founded in 1961 near Geneva."}]

    result = select_repair(trace, retrieve, analyzer=_analyzer())
    assert result["experiment"]["name"] == "merge_retrieved"
    assert result["recommendation"] == "ACCEPT EXPERIMENT"
    assert called == [2, 7]
    assert result["retriever_calls"] == 2
    assert {row["name"] for row in result["tried"]} == {"merge_retrieved", "rerun_retriever", "increase_k"}


def test_selective_abstains_when_the_repair_regresses():
    class LampJudge:
        def judge(self, claim, evidence, question=""):
            text = claim.lower()
            ev = evidence.lower()
            if "paris" in text:
                ok = "paris" in ev and "france" in ev
            elif "40 watts" in text:
                ok = "40 watts" in ev
            else:
                ok = False
            return JudgeResult("supported" if ok else "unsupported", "")

    trace = Trace(
        question="What is the capital of France?",
        answer="Paris is the capital of France. The desk lamp uses 40 watts.",
        retrieved_chunks=[Chunk(id="lamp", text="The desk lamp uses 40 watts.")],
        corpus_chunks=[
            Chunk(id="lamp", text="The desk lamp uses 40 watts.", rank=1),
            Chunk(id="paris", text="Paris is the capital of France.", rank=6),
        ],
        metadata=TraceMetadata(top_k=1),
    )

    def retrieve(query, k):
        if int(k) <= 1:
            return [{"id": "lamp", "text": "The desk lamp uses 40 watts."}]
        return [{"id": "paris", "text": "Paris is the capital of France."}]

    analyzer = TraceAnalyzer(FakeEmbedder(), LampJudge())
    guarded = select_repair(trace, retrieve, analyzer=analyzer)
    unguarded = select_repair(trace, retrieve, analyzer=analyzer, mode="no_guard")
    assert guarded["recommendation"] == "ABSTAIN"
    assert guarded["experiment"]["name"] == "abstain"
    assert guarded["regressions"] == 0
    assert guarded["retriever_calls"] == 2
    increase = next(row for row in guarded["tried"] if row["name"] == "increase_k")
    assert increase["regressions"] == 1
    assert unguarded["experiment"]["name"] == "increase_k"
    assert unguarded["regressions"] == 1


def test_selective_abstains_without_a_call_when_the_claim_is_unknown():
    trace = Trace(
        question="What is the capital of France?",
        answer="Paris is the capital of France.",
        retrieved_chunks=[Chunk(id="bananas", text="Bananas are a yellow fruit.")],
        metadata=TraceMetadata(top_k=1),
    )
    called = []

    def retrieve(query, k):
        called.append(k)
        return [{"id": "paris", "text": "Paris is the capital of France."}]

    result = select_repair(trace, retrieve, analyzer=_analyzer())
    assert result["recommendation"] == "ABSTAIN"
    assert called == []
    assert result["retriever_calls"] == 0


def test_dataset_label_and_budget_are_not_a_model_score():
    assert dataset_supported("Paris", "Paris is the capital of France.")
    assert not dataset_supported("Geneva", "Paris is the capital of France.")
    assert repair_under_budget(repairs=70, regressions=3, controls=100) == 70
    assert repair_under_budget(repairs=74, regressions=19, controls=100) is None
    with pytest.raises(ValueError):
        assert_not_frozen("holdout")
    summary = summarize_outcomes(
        [
            {"kind": "failure", "repaired": True, "abstained": False, "calls": 2},
            {"kind": "failure", "repaired": False, "abstained": True, "calls": 2},
            {"kind": "control", "regressed": False, "calls": 2},
            {"kind": "control", "regressed": True, "calls": 2},
        ]
    )
    assert summary["repair_success"] == 0.5
    assert summary["regression_rate"] == 0.5
    assert summary["repair_under_budget"] is None
