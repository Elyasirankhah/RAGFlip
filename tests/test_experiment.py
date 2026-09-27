from examples.my_retriever import retrieve
from rag.analyzer import TraceAnalyzer
from rag.diagnose import diagnose
from rag.experiment import choose_experiment, run_verified_repair
from rag.trace import Chunk, Trace, TraceMetadata
from tests.test_analyzer import FakeEmbedder, FakeJudge


def _analyzer():
    return TraceAnalyzer(FakeEmbedder(), FakeJudge())


def _trace():
    return Trace(
        question="What is the capital of France?",
        answer="Paris is the capital of France.",
        retrieved_chunks=[Chunk(id="r1", text="Bananas are a yellow fruit.")],
        corpus_chunks=[Chunk(id="c9", text="Paris is the capital of France.", rank=1, score=0.9)],
        metadata=TraceMetadata(top_k=1),
    )


def test_example_retriever_ranks_the_matching_file_first():
    library = retrieve("When did the Oak Street library open?", 1)
    cern = retrieve("When was CERN founded?", 1)
    assert library[0]["id"] == "library"
    assert cern[0]["id"] == "cern"


def test_rerun_verifies_the_failed_claim():
    calls = []

    def retrieve(query, k):
        calls.append((query, k))
        return [{"id": "gold", "text": "Paris is the capital of France."}]

    result = run_verified_repair(_trace(), retrieve, analyzer=_analyzer())
    assert calls == [("What is the capital of France?", 6)]
    assert result["experiment"]["name"] == "increase_k"
    assert result["claims"][0]["before"] == "retrieval_miss"
    assert result["claims"][0]["after"] == "supported"
    assert result["verified"] is True


def test_same_bad_chunks_are_not_verified():
    def retrieve(query, k):
        return [{"id": "r1", "text": "Bananas are a yellow fruit."}]

    result = run_verified_repair(_trace(), retrieve, analyzer=_analyzer())
    assert result["claims"][0]["after"] != "supported"
    assert result["verified"] is False


def test_chunking_is_not_pretended_as_a_retriever_call():
    trace = Trace(
        question="When was CERN founded?",
        answer="CERN was founded in 1961.",
        retrieved_chunks=[
            Chunk(id="left", text="CERN was founded"),
            Chunk(id="right", text="in 1961 near Geneva."),
        ],
        corpus_chunks=[
            Chunk(id="left", text="CERN was founded"),
            Chunk(id="right", text="in 1961 near Geneva."),
        ],
        metadata=TraceMetadata(top_k=2),
    )
    called = []

    def retrieve(query, k):
        called.append(k)
        return []

    result = run_verified_repair(trace, retrieve, analyzer=_analyzer())
    assert result["experiment"]["name"] == "merge_retrieved"
    assert result["experiment"]["runnable"] is True
    assert called == []
    assert result["claims"][0]["after"] == "supported"
    assert result["verified"] is True


def test_external_tfidf_retriever_widens_k_and_verifies():
    pytest = __import__("pytest")
    pytest.importorskip("sklearn")
    from examples.tfidf_retriever import retrieve as tfidf_retrieve
    from rag.analyzer import K_TOO_SMALL

    class TokenEmbedder:
        def embed_text(self, text):
            words = [word.lower().strip(".,?") for word in text.split()]
            return [float(len(words)), float(sum(len(word) for word in words))]

        def embed_batch(self, texts):
            return [self.embed_text(text) for text in texts]

    class ExactJudge:
        def judge(self, claim, evidence, question=""):
            from rag.judge import JudgeResult

            if claim.strip() and claim.strip() in evidence:
                return JudgeResult("supported", "exact passage")
            return JudgeResult("unsupported", "passage missing")

    question = "Where are the Apollo mission notebooks kept?"
    top = tfidf_retrieve(question, 1)
    assert top[0]["id"] == "catalog_notice.txt"
    assert tfidf_retrieve(question, 6)[1]["id"] == "special_collections.txt"

    trace = Trace.load("examples/traces/apollo_k_too_small.json")
    analyzer = TraceAnalyzer(TokenEmbedder(), ExactJudge())
    calls = []

    def counting_retrieve(query, k):
        calls.append((query, k))
        return tfidf_retrieve(query, k)

    result = run_verified_repair(trace, counting_retrieve, analyzer=analyzer)
    assert result["experiment"]["name"] == "increase_k"
    assert result["experiment"]["k"] == 6
    assert calls == [(question, 6)]
    assert result["claims"][0]["before"] == "retrieval_miss"
    assert result["before"]["claims"][0]["component"] == K_TOO_SMALL
    assert result["claims"][0]["after"] == "supported"
    assert result["verified"] is True


def test_choose_experiment_restores_the_original_query():
    trace = Trace(
        question="bananas",
        answer="Paris is the capital of France.",
        retrieved_chunks=[Chunk(id="r1", text="Bananas are a yellow fruit.")],
        corpus_chunks=[Chunk(id="c9", text="Paris is the capital of France.", rank=1, score=0.4)],
        metadata=TraceMetadata(top_k=1, original_query="What is the capital of France?", rewritten_query="bananas"),
    )
    before = diagnose(trace, analyzer=_analyzer())
    experiment = choose_experiment(trace, before)
    assert experiment["name"] == "restore_original_query"
    assert experiment["query"] == "What is the capital of France?"
