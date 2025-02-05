import numpy as np


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ssr = np.sum((y_true - y_pred) ** 2)
    sst = np.sum((y_true - np.mean(y_pred)) ** 2)
    return round(1 - (ssr / sst), 3)
