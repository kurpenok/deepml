import numpy as np


def dynamic_tanh(
    x: np.ndarray, alpha: float, gamma: np.ndarray, beta: np.ndarray
) -> np.ndarray:
    return np.round(gamma * np.tanh(alpha * x) + beta, 4)
