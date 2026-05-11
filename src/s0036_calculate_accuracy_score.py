import numpy as np


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.sum(y_true == y_pred) / y_true.shape[0]
