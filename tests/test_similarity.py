from rag.similarity import cosine_similarity


def test_identical_vectors():
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0


def test_orthogonal_and_zero_vectors():
    assert abs(cosine_similarity([1.0, 0.0], [0.0, 1.0])) < 1e-9
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0
