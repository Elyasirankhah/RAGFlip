from examples.wiki_demo import DemoIndex, gold_counts, logged_traces, select_questions


def test_selects_one_long_answer_per_passage():
    questions = [
        {"id": "a", "passage_id": "p0", "question": "first", "answer": "alpha beta gamma"},
        {"id": "b", "passage_id": "p0", "question": "second", "answer": "alpha beta gamma"},
        {"id": "c", "passage_id": "p1", "question": "third", "answer": "no"},
    ]
    picked = select_questions(questions, ["p0", "p1"], limit=10, seed=1)
    assert len(picked) == 1
    assert picked[0]["passage_id"] == "p0"
    assert picked[0]["id"] in {"a", "b"}


def test_gold_counts_fixed_and_broken():
    questions = [
        {"id": "1", "answer": "red rock"},
        {"id": "2", "answer": "blue lake"},
    ]
    before = {"1": "nothing here", "2": "the blue lake is deep"}
    after = {"1": "red rock dam", "2": "unrelated text"}
    assert gold_counts(questions, before, after) == {
        "failing": 1,
        "fixed": 1,
        "working": 1,
        "broken": 1,
    }


def test_logged_traces_keep_the_bm25_hit():
    passages = [
        {"id": "p0", "text": "The Cedar Span measures 240 meters across the gorge."},
        {"id": "p1", "text": "Bananas are a yellow fruit sold in markets."},
    ]
    questions = [
        {
            "id": "q",
            "passage_id": "p0",
            "question": "How long is the Cedar Span?",
            "answer": "240 meters across",
        }
    ]
    rows = logged_traces(questions, DemoIndex(passages), k=1)
    assert rows[0][0] == "q.json"
    assert rows[0][1].retrieved_chunks[0].id == "p0"
    assert rows[0][1].metadata.top_k == 1
