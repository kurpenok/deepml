import numpy as np


def precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))

    return 0 if tp + fp == 0 else tp / (tp + fp)


def recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    return 0 if tp + fn == 0 else tp / (tp + fn)


def f_score(y_true: np.ndarray, y_pred: np.ndarray, beta: float) -> float:
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)

    return round((1 + beta**2) * ((p * r) / ((beta**2 * p) + r)), 3)
