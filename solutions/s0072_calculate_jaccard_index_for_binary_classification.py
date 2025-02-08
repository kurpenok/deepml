import numpy as np


def jaccard_index(y_true: np.ndarray, y_pred: np.ndarray):
    intersection = np.logical_and(y_true, y_pred)
    union = np.logical_or(y_true, y_pred)
    return round(intersection.sum() / union.sum(), 3)
