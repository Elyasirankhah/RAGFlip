import json
from pathlib import Path

from rag.datasets.ragtruth import export_split, gold_label, iter_examples, question_of, score_examples, source_to_text
from rag.judge import _parse_judge_output
from rag.local_models import DEFAULT_EMBED_MODEL, DEFAULT_JUDGE_MODEL


def test_local_judge_defaults_and_json_parse():
    assert DEFAULT_JUDGE_MODEL == "Qwen/Qwen3.8-27B"
    assert DEFAULT_EMBED_MODEL == "BAAI/bge-large-en-v1.5"
    parsed = _parse_judge_output('{"label":"supported","reason":"paraphrase"}')
    assert parsed.label == "supported"
    assert parsed.reason == "paraphrase"


def test_gold_and_question_mapping():
    assert gold_label([]) == "supported"
    assert gold_label([{"label_type": "Evident Conflict"}]) == "hallucination"
    assert question_of("QA", {"question": "capital of France?"}) == "capital of France?"
    assert "Summarize" in question_of("Summary", "article")
    assert "phone" in source_to_text({"question": "q", "passages": "passage 1: phone 555"})


def test_export_joins_response_to_source(tmp_path: Path):
    root = tmp_path / "dataset"
    root.mkdir()
    (root / "source_info.jsonl").write_text(
        json.dumps(
            {
                "source_id": 1,
                "task_type": "QA",
                "source": "web",
                "source_info": {"question": "butcher shop phone number", "passages": "Phone Number: (510) 889-8690"},
                "prompt": "answer",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "response.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": 7,
                        "source_id": 1,
                        "model": "gpt-4-0613",
                        "temperature": 0,
                        "labels": [{"label_type": "Evident Baseless Info", "text": "open Sundays"}],
                        "split": "test",
                        "quality": "good",
                        "response": "The phone is (510) 889-8690. It is open Sundays.",
                    }
                ),
                json.dumps(
                    {
                        "id": 8,
                        "source_id": 1,
                        "model": "gpt-4-0613",
                        "temperature": 0,
                        "labels": [],
                        "split": "train",
                        "quality": "good",
                        "response": "The phone is (510) 889-8690.",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "test.jsonl"
    counts = export_split(root, out, "test")
    assert counts == {"supported": 0, "hallucination": 1, "total": 1}
    example = next(iter_examples(root, "test"))
    assert example["trace"]["question"] == "butcher shop phone number"
    assert example["trace"]["retrieved_chunks"][0]["text"].startswith("Phone Number")
    assert example["trace"]["metadata"]["gold"] == "hallucination"


def test_score_checkpoint_resumes(tmp_path: Path, monkeypatch):
    called = []

    def fake_predict(example, analyzer):
        called.append(example["trace"]["metadata"]["response_id"])
        return "supported"

    monkeypatch.setattr("rag.datasets.ragtruth.predict_response", fake_predict)
    examples = [
        {"gold": "hallucination", "trace": {"metadata": {"response_id": 1, "task_type": "QA"}}},
        {"gold": "supported", "trace": {"metadata": {"response_id": 2, "task_type": "QA"}}},
    ]
    checkpoint = tmp_path / "partial.jsonl"
    summary = tmp_path / "summary.json"
    checkpoint.write_text(
        json.dumps({"response_id": 1, "gold": "hallucination", "pred": "hallucination", "task_type": "QA"}) + "\n",
        encoding="utf-8",
    )
    report = score_examples(examples, analyzer=None, judge_name="local", checkpoint_path=checkpoint, summary_path=summary)
    assert called == [2]
    assert report["n"] == 2
    assert report["complete"] is True
    assert len(checkpoint.read_text(encoding="utf-8").splitlines()) == 2
    assert json.loads(summary.read_text(encoding="utf-8"))["n"] == 2
