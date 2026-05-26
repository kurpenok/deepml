import numpy as np


def f_score(y_true: np.ndarray, y_pred: np.ndarray, beta: float) -> float:
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0

    return (1 + beta**2) * ((precision * recall) / ((beta**2 * precision) + recall))
