import numpy as np

from src.s0071_calculate_root_mean_square_error import rmse


def test_case_1():
    y_true = np.array([3.0, -0.5, 2.0, 7.0])
    y_pred = np.array([2.5, 0.0, 2.0, 8.0])

    result = rmse(y_pred, y_true)
    expected = 0.6123724356957945

    assert result == expected


def test_case_2():
    y_true = np.array([[0.5, 1.0], [-1.0, 1.0], [7.0, -6.0]])
    y_pred = np.array([[0.0, 2.0], [-1.0, 2.0], [8.0, -5.0]])

    result = rmse(y_pred, y_true)
    expected = 0.8416254115301732

    assert result == expected


def test_case_3():
    y_true = np.array([[1.0, 2.0], [3.0, 4.0]])
    y_pred = np.array([[1.0, 2.0], [3.0, 4.0]])

    result = rmse(y_pred, y_true)
    expected = 0.0

    assert result == expected
