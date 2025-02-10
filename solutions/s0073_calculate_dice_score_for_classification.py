import numpy as np


def dice_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if not y_true.sum() or not y_pred.sum():
        return 0.0

    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    return round((2 * tp) / (2 * tp + fp + fn), 3)
