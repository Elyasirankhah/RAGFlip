"""Vector similarity helpers."""
from typing import List

import numpy as np


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    left = np.array(vec1, dtype=np.float64)
    right = np.array(vec2, dtype=np.float64)
    denom = np.linalg.norm(left) * np.linalg.norm(right)
    if denom == 0:
        return 0.0
    return float(np.dot(left, right) / denom)
