import numpy as np


def shuffle_data(
    X: np.ndarray, y: np.ndarray, seed=None
) -> tuple[np.ndarray, np.ndarray]:
    if seed:
        np.random.seed(seed)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    return X[indices], y[indices]
