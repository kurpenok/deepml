import numpy as np


def precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return 0.0 if tp + fp == 0 else tp / (tp + fp)


def recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return 0.0 if tp + fn == 0 else tp / (tp + fn)


def calculate_f1_score(y_true: list[int], y_pred: list[int]) -> float:
    p = precision(np.array(y_true), np.array(y_pred))
    r = recall(np.array(y_true), np.array(y_pred))
    return 0.0 if p + r == 0 else round(2 * ((p * r) / (p + r)), 3)
