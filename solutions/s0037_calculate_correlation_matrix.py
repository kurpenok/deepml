import numpy as np


def calculate_correlation_matrix(
    X: np.ndarray, Y: np.ndarray | None = None
) -> np.ndarray:
    Y = X if Y is None else Y

    X_centered = X - X.mean(axis=0, keepdims=True)
    Y_centered = Y - Y.mean(axis=0, keepdims=True)

    covariance = (X_centered.T @ Y_centered) / X.shape[0]

    std_X = np.std(X, axis=0, keepdims=True, ddof=0).T
    std_Y = np.std(Y, axis=0, keepdims=True, ddof=0)

    return covariance / (std_X @ std_Y)
