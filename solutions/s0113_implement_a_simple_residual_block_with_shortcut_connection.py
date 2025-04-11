import numpy as np


def residual_block(x: np.ndarray, w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
    return np.maximum(0, w2 @ np.maximum(0, w1 @ x) + x)
