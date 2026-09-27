import json
from pathlib import Path

from examples.hetero_retriever import retrieve
from rag.analyzer import TraceAnalyzer
from rag.experiment import execute_experiment, run_verified_repair
from rag.judge import JudgeResult
from rag.trace import Trace
from tests.test_analyzer import FakeEmbedder

ROOT = Path(__file__).resolve().parents[1] / "examples" / "traces" / "hetero"


class HeteroJudge:
    def judge(self, claim: str, evidence: str, question: str = "") -> JudgeResult:
        claim_l = claim.lower()
        evidence_l = evidence.lower()
        if "40 watts" in claim_l:
            if "uses 40 watts" in evidence_l and "400 watts" not in evidence_l:
                return JudgeResult("supported", "precise card")
            return JudgeResult("unsupported", "precise card missing")
        if "mars" in claim_l:
            return JudgeResult("unsupported", "not in the corpus")
        if "paris" in claim_l and "capital" in claim_l:
            if "paris" in evidence_l and "capital" in evidence_l and "france" in evidence_l:
                return JudgeResult("supported", "paris")
            return JudgeResult("unsupported", "paris missing")
        if "1961" in claim_l:
            if "founded" in evidence_l and "1961" in evidence_l:
                return JudgeResult("supported", "both halves")
            return JudgeResult("unsupported", "split")
        return JudgeResult("unsupported", "default")


def _analyzer():
    return TraceAnalyzer(FakeEmbedder(), HeteroJudge())


def _run(name: str):
    trace = Trace.load(ROOT / f"{name}.json")
    return run_verified_repair(trace, retrieve, analyzer=_analyzer())


def test_default_repair_is_wider_retrieval_when_that_reaches_the_passage():
    result = _run("wider_k_works")
    assert result["experiment"]["name"] == "increase_k"
    assert result["verified"] is True
    assert result["regressions"] == 0


def test_hallucination_is_not_sent_back_to_the_retriever():
    result = _run("hallucination")
    assert result["experiment"]["name"] == "no_retrieval_change"
    assert result["verified"] is False
    assert result["regressions"] == 0


def test_wider_retrieval_regresses_a_supported_claim_when_the_budget_drops_it():
    trace = Trace.load(ROOT / "wider_k_regresses.json")
    before = __import__("rag.diagnose", fromlist=["diagnose"]).diagnose(trace, analyzer=_analyzer())
    forced = {
        "name": "increase_k",
        "query": trace.question,
        "k": 6,
        "why": "baseline",
        "runnable": True,
    }
    result = execute_experiment(trace, retrieve, _analyzer(), forced, before)
    assert result["verified"] is False
    assert result["regressions"] == 1
    selected = run_verified_repair(trace, retrieve, analyzer=_analyzer())
    assert selected["experiment"]["name"] == "no_retrieval_change"
    assert selected["regressions"] == 0


def test_merge_repairs_a_split_passage_that_wider_retrieval_does_not():
    result = _run("merge_works")
    assert result["experiment"]["name"] == "merge_retrieved"
    assert result["verified"] is True


def test_original_query_repairs_a_bad_rewrite():
    result = _run("query_rewrite")
    assert result["experiment"]["name"] == "restore_original_query"
    assert result["verified"] is True


def test_logged_rerank_score_is_used_instead_of_wider_retrieval():
    result = _run("rerank")
    assert result["experiment"]["name"] == "rerank_candidates"
    assert result["verified"] is True


def test_supported_trace_runs_no_experiment():
    result = _run("supported")
    assert result["experiment"] is None
    assert result["verified"] is False


def test_dev_set_lists_every_failure_mode():
    expected = json.loads((ROOT / "expected.json").read_text(encoding="utf-8"))
    assert {row["trace"] for row in expected} == {
        "wider_k_works",
        "hallucination",
        "wider_k_regresses",
        "merge_works",
        "query_rewrite",
        "rerank",
        "supported",
    }
