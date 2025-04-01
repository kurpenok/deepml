import numpy as np


def pattern_weaver(n: int, crystal_values: np.ndarray, dimension: int) -> np.ndarray:
    scores = np.outer(crystal_values, crystal_values) / np.sqrt(dimension)

    max_scores = np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(scores - max_scores)
    softmax = exp_scores / exp_scores.sum(axis=1, keepdims=True)

    return np.round(softmax @ crystal_values, 4)
