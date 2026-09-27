from rag.chunker import chunk_text


def test_short_text_is_single_chunk():
    chunks = chunk_text("hello world", chunk_size=500, chunk_overlap=50)
    assert len(chunks) == 1
    assert chunks[0]["text"] == "hello world"
    assert chunks[0]["start_idx"] == 0


def test_long_text_produces_overlapping_chunks():
    words = " ".join(f"word{i:02d}" for i in range(40))
    chunks = chunk_text(words, chunk_size=40, chunk_overlap=10)
    assert len(chunks) > 1
    assert all("text" in chunk and "start_idx" in chunk for chunk in chunks)
    assert all(chunk["text"] for chunk in chunks)
    assert chunks[0]["start_idx"] == 0
    assert chunks[1]["start_idx"] > 0


def test_chunk_text_cleans_input():
    chunks = chunk_text("hello   \n  world", chunk_size=500, chunk_overlap=50)
    assert chunks[0]["text"] == "hello world"
