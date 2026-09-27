from rag.analyzer import CHUNKING_MISS, HALLUCINATION, RETRIEVAL_MISS, SUPPORTED, TraceAnalyzer
from rag.datasets.real_faults import (
    SEEN_SOURCE_IDS,
    build_dev_cases,
    build_holdout_cases,
    build_real_fault_cases,
    evaluate_with_repair,
    replay_chunks,
)
from tests.test_analyzer import FakeEmbedder, FakeJudge


def _sources():
    return [
        {
            "source_id": "france",
            "question": "What is the capital of France?",
            "passages": [
                "Paris is the capital of France and sits on the Seine. "
                "The city has many museums. The river is called the Seine. "
                "Visitors often see the old bridges. Bread is sold in the morning markets. "
                "Trains leave from several stations."
            ],
        },
        {
            "source_id": "cern",
            "question": "When was CERN founded?",
            "passages": [
                "CERN was founded in 1961 near the city of Geneva. "
                "Scientists from many countries work there. The machines fill large tunnels. "
                "Public tours run on selected days. Offices sit above the main site. "
                "The cafeteria opens early."
            ],
        },
    ]


def test_real_cases_use_source_text_and_one_break_each():
    cases = build_real_fault_cases(_sources(), limit=2)
    assert len(cases) == 8
    by_name = {case["name"]: case for case in cases}
    france_chunk = by_name["france-chunking_miss"]
    assert france_chunk["answer"] in france_chunk["corpus_chunks"][0]["text"] + " " + france_chunk["corpus_chunks"][1]["text"]
    assert "Seine" in france_chunk["answer"]
    retrieval = by_name["france-retrieval_miss"]
    retrieved_text = " ".join(chunk["text"] for chunk in retrieval["retrieved_chunks"])
    assert "capital of France" not in retrieved_text
    assert any("capital of France" in chunk["text"] for chunk in retrieval["corpus_chunks"])
    hallucination = by_name["france-hallucination"]
    assert "CERN" in hallucination["answer"]
    assert all("CERN" not in chunk["text"] for chunk in hallucination["retrieved_chunks"])


def test_replay_follows_the_predicted_stage():
    cases = {case["name"]: case for case in build_real_fault_cases(_sources(), limit=2)}
    merged = replay_chunks(cases["france-chunking_miss"], CHUNKING_MISS)
    assert merged is not None and "capital of France" in merged[0]["text"]
    retrieval = cases["france-retrieval_miss"]
    restored = replay_chunks(retrieval, RETRIEVAL_MISS, embedder=FakeEmbedder())
    assert restored is not None
    assert len(restored) == len(retrieval["retrieved_chunks"])
    assert len(restored) < len(retrieval["corpus_chunks"])
    assert any("capital of France" in chunk["text"] for chunk in restored)
    assert replay_chunks(cases["france-hallucination"], HALLUCINATION) is None
    assert replay_chunks(cases["france-supported"], SUPPORTED) is None


def test_dev_split_leaves_out_articles_already_used():
    sources = _sources() + [
        {"source_id": "15394", "question": "Seen already?", "passages": ["This article was already inspected."]}
    ]
    cases, dev_ids, holdout_ids = build_dev_cases(sources, limit=2, holdout_limit=1)
    assert "15394" in SEEN_SOURCE_IDS
    assert "15394" not in dev_ids
    assert "15394" not in holdout_ids
    assert set(dev_ids) <= {"france", "cern"}
    assert cases
    assert holdout_ids == [] or set(holdout_ids).isdisjoint(SEEN_SOURCE_IDS)


def test_holdout_builder_keeps_only_the_requested_articles():
    cases, missing = build_holdout_cases(_sources() + [
        {"source_id": "15394", "question": "Seen already?", "passages": ["This article was already inspected."]}
    ], ["france", "missing", "15394"])
    names = {case["name"].split("-")[0] for case in cases}
    assert names == {"france"}
    assert "missing" in missing
    assert "15394" in missing


def test_checkpoint_skips_cases_already_saved(tmp_path):
    cases = build_real_fault_cases(_sources(), limit=2)
    checkpoint = tmp_path / "partial.jsonl"
    analyzer = TraceAnalyzer(FakeEmbedder(), FakeJudge())
    first = evaluate_with_repair(analyzer, cases[:1], checkpoint=checkpoint)
    assert first["total"] == 1
    calls = {"n": 0}
    original = analyzer.analyze

    def counting_analyze(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    analyzer.analyze = counting_analyze
    second = evaluate_with_repair(analyzer, cases[:1], checkpoint=checkpoint)
    assert calls["n"] == 0
    assert second["cases"][0]["name"] == first["cases"][0]["name"]


def test_fake_judge_names_the_break_and_the_replay_fixes_it():
    cases = build_real_fault_cases(_sources(), limit=2)
    report = evaluate_with_repair(TraceAnalyzer(FakeEmbedder(), FakeJudge()), cases)
    assert report["rejected_ambiguous"] == 1
    assert report["correct"] == report["total"]
    assert report["repair_attempted"] == 3
    assert report["repair_improved"] == 3
    assert report["end_to_end_rate"] == 1.0
    assert report["macro_f1"] == 1.0
