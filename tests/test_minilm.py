import pytest

pytest.importorskip('torch')
pytest.importorskip('safetensors')
pytest.importorskip('tokenizers')

import numpy as np

from examples.minilm import from_cache


def test_minilm_ranks_a_related_sentence_closer():
    model = from_cache()
    vectors = model.encode(
        [
            "Paris is the capital of France.",
            "The capital city of France is Paris.",
            "Bananas are a yellow fruit.",
        ]
    )
    paris, capital, bananas = vectors
    assert np.dot(paris, capital) > np.dot(paris, bananas)
