from rag_debugger.adapters import from_langchain, from_llama_index, from_openinference, ingest


class FakeDocument:
    def __init__(self, text, metadata=None):
        self.page_content = text
        self.metadata = metadata or {"id": "doc-1"}


class FakeLlamaResponse:
    def __init__(self):
        self.response = "Paris is the capital of France."
        self.source_nodes = [type("N", (), {"text": "Paris is the capital of France.", "metadata": {"id": "n1"}, "node": None})()]


def test_from_langchain_result_dict():
    trace = from_langchain(
        "What is the capital of France?",
        {"result": "Paris is the capital of France.", "source_documents": [FakeDocument("Paris is the capital of France.")]},
    )
    assert trace["answer"].startswith("Paris")
    assert trace["retrieved_chunks"][0]["text"].startswith("Paris")


def test_from_llama_index_and_openinference():
    llama = from_llama_index("q", FakeLlamaResponse())
    assert llama["retrieved_chunks"][0]["text"]

    span = {
        "attributes": {
            "input.value": "What is the capital of France?",
            "output.value": "Paris is the capital of France.",
            "retrieval.documents": [{"id": "r1", "text": "Paris is the capital of France."}],
        }
    }
    oi = from_openinference(span)
    assert oi["question"].startswith("What")
    ingested = ingest({"answer": "a", "retrieved_chunks": [{"id": "1", "text": "t"}]})
    assert ingested["answer"] == "a"
