import numpy as np


def layer_normalization(
    X: np.ndarray, gamma: np.ndarray, beta: np.ndarray, epsilon: float = 1e-5
) -> np.ndarray:
    return np.round(
        gamma
        * (X - np.mean(X, axis=-1, keepdims=True))
        / np.sqrt(np.var(X, axis=-1, keepdims=True) + epsilon)
        + beta,
        3,
    )
