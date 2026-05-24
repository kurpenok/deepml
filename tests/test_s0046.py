import numpy as np

from src.s0046_implement_precision_metric import precision


def test_case_1():
    y_true = np.array([1, 0, 1, 1, 0, 1])
    y_pred = np.array([1, 0, 1, 0, 0, 1])

    result = precision(y_true, y_pred)
    expected = 1.0

    assert result == expected


def test_case_2():
    y_true = np.array([1, 0, 1, 1, 0, 0])
    y_pred = np.array([1, 0, 0, 0, 0, 1])

    result = precision(y_true, y_pred)
    expected = 0.5

    assert result == expected
