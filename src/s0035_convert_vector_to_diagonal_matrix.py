import numpy as np


def make_diagonal(X: np.ndarray) -> np.ndarray:
    return np.array(
        [[X[i] if i == j else 0 for j in range(X.shape[0])] for i in range(X.shape[0])]
    )
