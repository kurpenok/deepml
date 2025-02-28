import numpy as np


def sigmoid(z: float) -> float:
    return 1 / (1 + np.exp(-z))


def custom_round(y: float) -> int:
    i, f = divmod(y, 1)
    return int(i + ((f >= 0.5) if (y > 0) else (f > 0.5)))


def predict_logistic(X: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
    return np.vectorize(custom_round)(np.vectorize(sigmoid)(X @ weights + bias))
