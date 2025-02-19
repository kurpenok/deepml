import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return round(np.mean(np.abs(y_true - y_pred)), 3)
