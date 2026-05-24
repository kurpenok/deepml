import numpy as np


def kernel_function(x1: np.ndarray, x2: np.ndarray) -> float:
    return np.sum(x1 * x2)
