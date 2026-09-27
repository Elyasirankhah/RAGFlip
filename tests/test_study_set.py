import json
from pathlib import Path

from rag.pipelines.study_set import build_records, load_study_cases, write_study
from rag.study import ENVIRONMENTS, assert_not_frozen, dataset_supported


def _rows():
    passages = []
    questions = []
    for index in range(40):
        passage_id = f"p{index}"
        answer = f"alpha{index}"
        passages.append({"id": passage_id, "text": f"The stored fact is {answer} in passage {index}.", "title": "t"})
        questions.append(
            {
                "id": f"q{index}",
                "passage_id": passage_id,
                "question": f"Where is {answer} stored?",
                "answer": answer,
            }
        )
    return questions, passages


def test_study_records_are_labeled_by_the_answer_string(tmp_path):
    questions, passages = _rows()
    records = build_records(questions, passages)
    assert records
    assert {record["environment"] for record in records} <= set(ENVIRONMENTS)
    manifest = write_study(records, passages, tmp_path)
    assert manifest["method_scored"] is False
    assert manifest["sealed_before_scoring"] is True
    dev = set(json.loads((tmp_path / "dev_ids.json").read_text(encoding="utf-8")))
    test = set(json.loads((tmp_path / "test_ids.json").read_text(encoding="utf-8")))
    assert dev.isdisjoint(test)
    assert dev | test == {record["name"] for record in records}
    cases = load_study_cases(tmp_path)
    for case in cases:
        text = "\n".join(chunk.text for chunk in case["trace"].retrieved_chunks)
        kind = "control" if dataset_supported(case["trace"].answer, text) else "failure"
        assert case["kind"] == kind
        again = case["retriever"](case["trace"].question, case["trace"].metadata.top_k)
        assert [chunk["id"] for chunk in again] == [chunk.id for chunk in case["trace"].retrieved_chunks]


def test_development_score_stays_on_the_dev_split(tmp_path):
    from rag.study_eval import score_development

    questions, passages = _rows()
    records = build_records(questions, passages)
    write_study(records, passages, tmp_path)
    summary = score_development(root=tmp_path, checkpoint=tmp_path / "score.jsonl")
    dev = json.loads((tmp_path / "dev_ids.json").read_text(encoding="utf-8"))
    scored = [json.loads(line)["name"] for line in (tmp_path / "score.jsonl").read_text(encoding="utf-8").splitlines()]
    assert summary["final_test_scored"] is False
    assert summary["cases"] == len(dev)
    assert set(scored) == set(dev)
    assert "selective" in summary
    assert "oracle" in summary


def test_sealed_squad_split_is_unscored():
    root = Path(__file__).resolve().parents[1] / "rag" / "datasets" / "study"
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    dev = json.loads((root / "dev_ids.json").read_text(encoding="utf-8"))
    test = json.loads((root / "test_ids.json").read_text(encoding="utf-8"))
    assert manifest["source"] == "SQuAD v1.1 dev"
    assert manifest["method_scored"] is False
    assert manifest["sealed_before_scoring"] is True
    assert manifest["failures"] >= 200
    assert manifest["controls"] >= 200
    assert set(dev).isdisjoint(test)
    assert len(dev) == manifest["dev"]
    assert len(test) == manifest["test"]


def test_secondary_budgets_do_not_move_the_primary():
    from rag.adjudicate import is_ambiguous
    from rag.study import REGRESSION_BUDGET, STUDY_JUDGE_MODELS, repair_under_budget

    assert REGRESSION_BUDGET == 0.05
    assert repair_under_budget(382, 95, 3972) == 382
    assert repair_under_budget(382, 95, 3972, 0.02) is None
    assert repair_under_budget(382, 95, 3972, 0.0) is None
    assert len(STUDY_JUDGE_MODELS) == 6
    assert is_ambiguous("May", "The Mayor spoke on Monday.")
    assert not is_ambiguous("Paris", "Paris is the capital of France.")
    assert not is_ambiguous("Paris", "Lyon is a city in France.")


def test_frozen_sets_stay_out_of_this_study():
    for name in ("holdout", "mixed-hotpot", "judge-dev"):
        try:
            assert_not_frozen(name)
        except ValueError:
            continue
        raise AssertionError(name)
