from rag.datasets.detector import detector_messages, parse_detector_label


def test_parse_detector_label():
    assert parse_detector_label('{"label":"supported"}') == "supported"
    assert parse_detector_label('{"label":"hallucination"}') == "hallucination"
    assert parse_detector_label("not json") == "hallucination"


def test_detector_training_message_uses_gold():
    example = {
        "gold": "supported",
        "trace": {
            "question": "phone?",
            "answer": "555",
            "retrieved_chunks": [{"id": "s", "text": "Phone 555"}],
            "metadata": {"response_id": 1},
        },
    }
    messages = detector_messages(example, with_label=True)
    assert messages[-1]["role"] == "assistant"
    assert "supported" in messages[-1]["content"]
    assert "Phone 555" in messages[1]["content"]
