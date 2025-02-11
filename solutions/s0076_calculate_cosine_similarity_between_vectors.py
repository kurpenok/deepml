import numpy as np


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    return round(float((v1 @ v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))), 3)
