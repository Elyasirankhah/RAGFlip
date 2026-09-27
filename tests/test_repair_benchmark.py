from rag.analyzer import (
    HALLUCINATION,
    K_TOO_SMALL,
    RETRIEVAL_MISS,
    TraceAnalyzer,
)
from rag.benchmark import evaluate_analyzer, planted_cases
from rag.repair import run_repair_experiments
from tests.test_analyzer import FakeEmbedder, FakeJudge


class RankEmbedder:
    """Lets tests control query-vs-gold ranking."""

    def embed_text(self, text: str):
        lowered = text.lower()
        if lowered.startswith("what is"):
            return [1.0, 0.0]
        if "paris is the capital of france" in lowered:
            return [0.15, 1.0]
        if "weather" in lowered or "unrelated" in lowered:
            return [0.95, 0.05]
        if "banana" in lowered or "mars" in lowered:
            return [0.0, 0.2]
        return FakeEmbedder().embed_text(text)

    def embed_batch(self, texts):
        return [self.embed_text(text) for text in texts]


def test_k_too_small_when_gold_is_just_below_k():
    analyzer = TraceAnalyzer(RankEmbedder(), FakeJudge())
    gold = {"id": "gold", "text": "Paris is the capital of France.", "rank": 8, "score": 0.2}
    fillers = [{"id": f"d{i}", "text": f"Unrelated weather note number {i}."} for i in range(8)]
    report = analyzer.analyze(
        question="What is the capital of France?",
        answer="Paris is the capital of France.",
        retrieved_chunks=fillers[:5],
        corpus_chunks=fillers + [gold],
    )
    assert report.sentences[0].fault == RETRIEVAL_MISS
    assert report.sentences[0].component == K_TOO_SMALL
    assert report.sentences[0].evidence_rank and report.sentences[0].evidence_rank > 5
    assert report.suggested_fixes


def test_repair_improves_supported_pct_when_gold_is_in_corpus():
    analyzer = TraceAnalyzer(FakeEmbedder(), FakeJudge())
    gold = {"id": "gold", "text": "Paris is the capital of France."}
    fillers = [{"id": f"d{i}", "text": f"Bananas are a yellow fruit {i}."} for i in range(5)]
    result = run_repair_experiments(
        analyzer,
        question="What is the capital of France?",
        answer="Paris is the capital of France.",
        retrieved_chunks=fillers,
        corpus_chunks=fillers + [gold],
    )
    assert result["experiments"][0]["supported_pct"] == 0
    assert result["best"]["supported_pct"] == 100
    assert result["delta_supported_pct"] == 100


def test_expanded_planted_set_covers_each_family():
    from rag.benchmark import RETRIEVAL_MISS, build_expanded_cases

    cases = build_expanded_cases()
    assert len(cases) == 48
    causes = {case["true_cause"] for case in cases}
    assert causes == {"supported", "hallucination", "chunking_miss", RETRIEVAL_MISS}


def test_fault_injection_benchmark_recovers_planted_causes():
    analyzer = TraceAnalyzer(RankEmbedder(), FakeJudge())
    # Use the cases whose labels FakeJudge + RankEmbedder can actually recover.
    cases = [case for case in planted_cases() if case["name"] in {"k_too_small", "chunking_miss", "hallucination"}]
    result = evaluate_analyzer(analyzer, cases)
    assert result["correct"] == result["total"]


def test_hallucination_component_stays_hallucination():
    analyzer = TraceAnalyzer(FakeEmbedder(), FakeJudge())
    report = analyzer.analyze(
        question="What is the capital of France?",
        answer="Bananas grow on Mars.",
        retrieved_chunks=[{"id": "r1", "text": "Paris is the capital of France."}],
        corpus_chunks=[{"id": "r1", "text": "Paris is the capital of France."}],
    )
    assert report.root_cause == HALLUCINATION
    assert report.sentences[0].fault == HALLUCINATION
