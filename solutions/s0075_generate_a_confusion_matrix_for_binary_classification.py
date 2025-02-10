import numpy as np


def confusion_matrix(data: list[list[int]]) -> list[list[int]]:
    y_true = np.array([pair[0] for pair in data])
    y_pred = np.array([pair[1] for pair in data])

    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    return [[tp, fn], [fp, tn]]
