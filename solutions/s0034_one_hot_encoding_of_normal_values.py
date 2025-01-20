import numpy as np


def to_categorical(X: np.ndarray, n_col: int | None = None):
    n_col = np.max(X) + 1 if n_col is None else n_col

    ohe = np.zeros((X.shape[0], n_col))
    ohe[np.arange(X.shape[0]), X] = 1

    return ohe
