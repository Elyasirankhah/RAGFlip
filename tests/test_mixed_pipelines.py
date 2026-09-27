import json
from pathlib import Path

from rag.analyzer import TraceAnalyzer
from rag.pipelines.mixed import REPAIR_OBSERVED, load_mixed_cases
from rag.pipelines.sparse import SparseIndex
from rag.repair_compare import compare_mixed
from rag.trace import Trace
from tests.test_analyzer import FakeEmbedder, FakeJudge


def test_budget_packer_drops_a_long_hit_when_k_grows():
    chunks = [
        {"id": "gold", "text": "capital france " + " ".join(["history"] * 120)},
        {"id": "s1", "text": "france catalog"},
        {"id": "s2", "text": "france notice"},
        {"id": "s3", "text": "france ledger"},
        {"id": "s4", "text": "france room"},
        {"id": "s5", "text": "france date"},
    ]
    index = SparseIndex(chunks)
    assert [hit["id"] for hit in index.budget("capital france", 1)] == ["gold"]
    wider = [hit["id"] for hit in index.budget("capital france", 6)]
    assert "gold" not in wider
    assert wider


def test_mixed_compare_calls_the_pipeline_retriever():
    calls = []

    def retrieve(query, k):
        calls.append((query, int(k)))
        return [{"id": "gold", "text": "Paris is the capital of France."}]

    trace = Trace.from_dict(
        {
            "question": "What is the capital of France?",
            "answer": "Paris is the capital of France.",
            "retrieved_chunks": [{"id": "bad", "text": "Bananas are a yellow fruit."}],
            "corpus_chunks": [
                {"id": "bad", "text": "Bananas are a yellow fruit.", "rank": 1, "score": 0.2},
                {"id": "gold", "text": "Paris is the capital of France.", "rank": 6, "score": 0.9},
            ],
            "metadata": {"top_k": 1, "retriever": "bm25"},
        }
    )
    report = compare_mixed(
        TraceAnalyzer(FakeEmbedder(), FakeJudge()),
        [
            {
                "name": "sample",
                "observed": "wider_k_reaches",
                "pipeline": "bm25",
                "trace": trace,
                "retriever": retrieve,
            }
        ],
        suggester=lambda _trace: "no_retrieval_change",
    )
    assert calls
    assert report["verified_repair_rate"]["ragfix"]["verified"] == 1
    assert "observed" not in trace.to_dict()


def test_hotpot_mix_is_pipeline_output_outside_the_holdout():
    cases = load_mixed_cases()
    failures = [case for case in cases if case["observed"] in REPAIR_OBSERVED]
    assert 30 <= len(failures) <= 50
    pipelines = {case["pipeline"] for case in cases}
    assert 3 <= len(pipelines) <= 5
    observed = {case["observed"] for case in cases}
    assert {"chunk_split", "query_rewrite", "rerank_logged", "wider_k_reaches", "wider_k_misses"} <= observed
    holdout_path = Path(__file__).resolve().parents[1] / "rag" / "datasets" / "holdout_ids.json"
    holdout = set(json.loads(holdout_path.read_text(encoding="utf-8"))["source_ids"])
    for case in cases:
        source_id = case["trace"].metadata.extra["source_id"]
        assert source_id not in holdout
        assert case["trace"].metadata.extra["source"] == "hotpotqa/hotpot_qa validation"
        again = [hit["id"] for hit in case["retriever"](case["trace"].question, case["trace"].metadata.top_k)]
        assert again == [chunk.id for chunk in case["trace"].retrieved_chunks]
