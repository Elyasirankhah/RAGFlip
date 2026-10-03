from rag.experiment import format_repair
from rag.nli_judge import NLISupportJudge
from rag.judge import JudgeResult


def test_format_repair_rejects_on_regression():
    text = format_repair(
        {
            "experiment": {"name": "increase_k", "k": 6, "why": "try wider k", "runnable": True},
            "failed_claims": 1,
            "failed_claims_improved": 1,
            "regressions": 1,
            "verified": False,
            "claims": [{"before": "supported", "after": "unsupported", "claim": "The lamp uses 40 watts."}],
        }
    )
    assert "Recommendation" in text
    assert "REJECT EXPERIMENT" in text
    assert "1 previously supported claim checked" in text
    assert "1 regression" in text
    assert "judge_supported=false" in text
    assert "Verified" not in text


def test_format_repair_accepts_when_verified():
    text = format_repair(
        {
            "experiment": {"name": "increase_k", "k": 6, "why": "try wider k", "runnable": True},
            "failed_claims": 1,
            "failed_claims_improved": 1,
            "regressions": 0,
            "verified": True,
            "claims": [{"before": "retrieval_miss", "after": "supported", "claim": "Paris is the capital of France."}],
        }
    )
    assert "ACCEPT EXPERIMENT" in text
    assert "judge_supported=true" in text
    assert "Likely retrieval failure" in text
    assert "Verified" not in text


def test_format_repair_is_uncertain_when_nothing_runs():
    text = format_repair(
        {
            "experiment": {
                "name": "no_retrieval_change",
                "k": 4,
                "from_k": 4,
                "why": "absent",
                "runnable": False,
            },
            "failed_claims": 1,
            "failed_claims_improved": 0,
            "regressions": 0,
            "verified": False,
            "claims": [{"before": "hallucination", "after": "not_rerun", "claim": "Bananas grow on Mars."}],
        }
    )
    assert "UNCERTAIN" in text
    assert "Not run" in text


def test_overlap_demo_accepts_and_rejects():
    from examples.demo_retriever import retrieve
    from rag.experiment import run_verified_repair
    from rag.judge_eval import make_judge_analyzer
    from rag.trace import Trace

    analyzer = make_judge_analyzer("overlap")
    accepted = run_verified_repair(
        Trace.load("examples/traces/accept_increase_k.json"),
        retrieve,
        analyzer=analyzer,
    )
    rejected = run_verified_repair(
        Trace.load("examples/traces/reject_increase_k.json"),
        retrieve,
        analyzer=analyzer,
    )
    assert accepted["recommendation"] == "ACCEPT EXPERIMENT"
    assert accepted["experiment"]["name"] == "increase_k"
    assert rejected["recommendation"] == "REJECT EXPERIMENT"
    assert rejected["regressions"] == 1


def test_langchain_retriever_hook_passes_k():
    from ragflip.integrations.langchain import as_retriever

    class FakeRetriever:
        def __init__(self):
            self.search_kwargs = {"k": 2}
            self.seen = None

        def invoke(self, query):
            self.seen = (query, self.search_kwargs["k"])
            return [{"id": "a", "page_content": "Paris is the capital of France."}]

    retriever = FakeRetriever()
    chunks = as_retriever(retriever)("capital of France", 6)
    assert retriever.seen == ("capital of France", 6)
    assert chunks == [{"id": "a", "text": "Paris is the capital of France."}]


def test_nli_maps_entailment_labels():
    class FakePipe:
        def __call__(self, _payload, top_k=None):
            return [
                {"label": "entailment", "score": 0.91},
                {"label": "neutral", "score": 0.07},
                {"label": "contradiction", "score": 0.02},
            ]

    judge = NLISupportJudge(model_name="fake-nli", pipe=FakePipe())
    result = judge.judge("Paris is the capital of France.", "Paris is the capital of France.")
    assert isinstance(result, JudgeResult)
    assert result.label == "supported"
