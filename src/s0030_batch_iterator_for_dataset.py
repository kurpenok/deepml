import numpy as np


def batch_iterator(
    X: np.ndarray, y: np.ndarray | None = None, batch_size: int = 64
) -> list[list[list[float]]]:
    batches = []

    for i in np.arange(0, X.shape[0], batch_size):
        begin, end = i, min(i + batch_size, X.shape[0])
        if y is not None:
            batches.append([X[begin:end], y[begin:end]])
        else:
            batches.append(X[begin:end])

    return batches
