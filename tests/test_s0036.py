import numpy as np

from src.s0036_calculate_accuracy_score import accuracy_score


def test_case_1():
    y_true = np.array([1, 0, 1, 1, 0, 1])
    y_pred = np.array([1, 0, 0, 1, 0, 1])

    result = accuracy_score(y_true, y_pred)
    expected = 0.8333333333333334

    assert result == expected


def test_case_2():
    y_true = np.array([1, 1, 1, 1])
    y_pred = np.array([1, 0, 1, 0])

    result = accuracy_score(y_true, y_pred)
    expected = 0.5

    assert result == expected
