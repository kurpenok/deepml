import numpy as np


def log_softmax(scores: np.ndarray) -> np.ndarray:
    max_score = np.max(scores)
    shifted = scores - max_score
    log_sum_exp = np.log(np.sum(np.exp(shifted)))
    result = shifted - log_sum_exp
    return np.round(result, 4)
