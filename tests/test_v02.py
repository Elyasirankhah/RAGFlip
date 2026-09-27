import json
from pathlib import Path

from rag.diagnose import diagnose
from rag.report import compare_reports
from rag.trace import Trace
from rag_debugger.cli import run_compare
from rag_debugger.integrations.langchain import trace
from tests.test_analyzer import FakeEmbedder, FakeJudge
from rag.analyzer import TraceAnalyzer


def test_trace_round_trip(tmp_path: Path):
    raw = {
        "question": "What is the capital of France?",
        "answer": "Paris is the capital of France.",
        "retrieved_chunks": [{"id": "r1", "text": "Bananas are yellow."}],
        "corpus_chunks": [{"id": "c47", "text": "Paris is the capital of France."}],
        "metadata": {"retriever": "faiss", "top_k": 5, "model": "gpt-4o-mini"},
    }
    path = tmp_path / "trace.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    loaded = Trace.load(path)
    assert loaded.metadata.retriever == "faiss"
    assert loaded.metadata.top_k == 5
    assert loaded.to_dict()["retrieved_chunks"][0]["id"] == "r1"


def test_langchain_adapter_returns_trace():
    result = trace(
        "What is the capital of France?",
        {"result": "Paris is the capital of France.", "source_documents": [{"id": "r1", "text": "Paris."}]},
        retriever="faiss",
        top_k=5,
        model="gpt-4o-mini",
    )
    assert isinstance(result, Trace)
    assert result.metadata.top_k == 5
    assert result.retrieved_chunks[0].id == "r1"


def test_diagnose_report_shape():
    analyzer = TraceAnalyzer(FakeEmbedder(), FakeJudge())
    loaded = Trace.from_dict(
        {
            "question": "What is the capital of France?",
            "answer": "Paris is the capital of France.",
            "retrieved_chunks": [{"id": "r1", "text": "Bananas are yellow."}],
            "corpus_chunks": [{"id": "c47", "text": "Paris is the capital of France."}],
        }
    )
    report = diagnose(loaded, analyzer=analyzer)
    claim = report.claims[0].to_dict()
    assert set(claim) >= {
        "claim",
        "label",
        "confidence",
        "best_retrieved_chunk",
        "best_corpus_chunk",
        "evidence",
        "reason",
    }
    assert claim["label"] == "retrieval_miss"
    assert claim["best_corpus_chunk"] == "c47"
    assert report.pipeline.primary_failure == "retrieval"
    assert "Test next:" in report.format()


def test_compare_saved_reports(tmp_path: Path):
    before = {
        "claims": [],
        "pipeline": {
            "supported": 1,
            "total": 2,
            "retrieval_misses": 1,
            "hallucinations": 0,
            "primary_failure": "retrieval",
        },
    }
    after = {
        "claims": [],
        "pipeline": {
            "supported": 2,
            "total": 2,
            "retrieval_misses": 0,
            "hallucinations": 0,
            "primary_failure": "none",
        },
    }
    text = compare_reports(before, after)
    assert "50.0% -> 100.0%" in text
    left = tmp_path / "before.json"
    right = tmp_path / "after.json"
    left.write_text(json.dumps(before), encoding="utf-8")
    right.write_text(json.dumps(after), encoding="utf-8")
    assert "50.0% -> 100.0%" in run_compare(str(left), str(right))
