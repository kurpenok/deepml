import numpy as np


def ridge_loss(X: np.ndarray, w: np.ndarray, y_true: np.ndarray, alpha: float) -> float:
    return sum([(y - y_hat) ** 2 for y, y_hat in zip(y_true, X @ w)]) / X.shape[
        0
    ] + alpha * sum([weight**2 for weight in w])
