import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return round(np.sqrt(np.mean((y_true - y_pred) ** 2)), 3)
