import pytest

from rag.debugger import RAGDebugger


def _debugger():
    return RAGDebugger(embedding_generator=None, similarity_threshold=0.7)


def test_cosine_similarity_identical_vectors():
    score = _debugger().compute_cosine_similarity([1.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    assert score == pytest.approx(1.0, abs=1e-6)


def test_cosine_similarity_orthogonal_vectors():
    score = _debugger().compute_cosine_similarity([1.0, 0.0], [0.0, 1.0])
    assert score == pytest.approx(0.0, abs=1e-6)


def test_cosine_similarity_zero_vector():
    assert _debugger().compute_cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0
