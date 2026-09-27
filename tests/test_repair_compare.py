from rag.analyzer import TraceAnalyzer
from rag.repair_compare import compare_repairs, repairable_cases, scored_names
from tests.test_analyzer import FakeJudge


class LowRankEmbedder:
    def embed_text(self, text):
        lowered = text.lower()
        if "paris" in lowered and "capital" in lowered:
            return [0.0, 1.0]
        return [1.0, 0.0]

    def embed_batch(self, texts):
        return [self.embed_text(text) for text in texts]


def _case():
    wrong = [{"id": f"d{i}", "text": f"Bananas are a yellow fruit {i}."} for i in range(5)]
    gold = {"id": "gold", "text": "Paris is the capital of France."}
    return {
        "name": "france-retrieval_miss",
        "true_cause": "retrieval_miss",
        "question": "What is the capital of France?",
        "answer": "Paris is the capital of France.",
        "retrieved_chunks": [wrong[0]],
        "corpus_chunks": wrong + [gold],
    }


def test_increase_k_verifies_when_the_same_k_rerun_misses():
    report = compare_repairs(
        TraceAnalyzer(LowRankEmbedder(), FakeJudge()),
        [_case()],
        suggester=lambda trace: "increase_k",
    )
    rates = report["verified_repair_rate"]
    assert rates["always_increase_k"]["verified"] == 1
    assert rates["llm_choice"]["verified"] == 1
    assert rates["always_rerun"]["verified"] == 0
    assert rates["rag_debugger"]["verified"] == 1


def test_scored_checkpoint_selects_repairable_cases(tmp_path):
    checkpoint = tmp_path / "holdout.jsonl"
    checkpoint.write_text(
        "\n".join(
            [
                '{"kind":"scored","name":"a-retrieval_miss","row":{"true_cause":"retrieval_miss"}}',
                '{"kind":"rejected","name":"a-chunking_miss"}',
                '{"kind":"scored","name":"a-supported","row":{"true_cause":"supported"}}',
            ]
        ),
        encoding="utf-8",
    )
    cases = [
        {"name": "a-retrieval_miss", "true_cause": "retrieval_miss"},
        {"name": "a-chunking_miss", "true_cause": "chunking_miss"},
        {"name": "a-supported", "true_cause": "supported"},
    ]
    selected = repairable_cases(cases, scored_names(checkpoint))
    assert [case["name"] for case in selected] == ["a-retrieval_miss"]
