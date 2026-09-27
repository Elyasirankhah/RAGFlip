from rag.systemone import LayaSupportJudge, SystemOneClient, choose_repair, claim_regressed
from rag.judge import JudgeResult


class FakeRouter:
    def __init__(self, answers):
        self.answers = answers
        self.calls = []

    def predict(self, state, questions, **kwargs):
        self.calls.append({"state": state, "questions": questions, "kwargs": kwargs})
        return {"answers": self.answers}


def test_laya_verifier_accepts_high_noul():
    client = SystemOneClient(backend="local", router=FakeRouter({"supported": {"noul": 0.91}}))
    judge = LayaSupportJudge(client=client, threshold=0.5)
    result = judge.judge("Paris is the capital of France.", "Paris is the capital of France.")
    assert isinstance(result, JudgeResult)
    assert result.label == "supported"
    assert "0.910" in result.reason


def test_laya_verifier_rejects_low_noul():
    client = SystemOneClient(backend="local", router=FakeRouter({"supported": {"noul": 0.12}}))
    judge = LayaSupportJudge(client=client, threshold=0.5)
    result = judge.judge("Bananas grow on Mars.", "Paris is the capital of France.")
    assert result.label == "unsupported"


def test_choose_repair_returns_typed_choice():
    client = SystemOneClient(
        backend="local",
        router=FakeRouter({"repair": {"choice": "restore_original_query", "confidence": 0.8}}),
    )
    assert choose_repair(client, "q", "a", "chunks") == "restore_original_query"


def test_regression_guard_reads_noul():
    client = SystemOneClient(backend="local", router=FakeRouter({"damaged": {"noul": 0.88}}))
    assert claim_regressed(client, "claim", "before", "after") is True
    client = SystemOneClient(backend="local", router=FakeRouter({"damaged": {"noul": 0.1}}))
    assert claim_regressed(client, "claim", "before", "after") is False
