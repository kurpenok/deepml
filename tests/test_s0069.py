import numpy as np

from src.s0069_calculate_r_squared_for_regression_analysis import r_squared


def test_case_1():
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1, 2, 3, 4, 5])

    result = r_squared(y_true, y_pred)
    expected = 1.0

    assert result == expected


def test_case_2():
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1.1, 2.1, 2.9, 4.2, 4.8])

    result = r_squared(y_true, y_pred)
    expected = 0.989

    assert result == expected


def test_case_3():
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([2, 1, 4, 3, 5])

    result = r_squared(y_true, y_pred)
    expected = 0.6

    assert result == expected


def test_case_4():
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([3, 3, 3, 3, 3])

    result = r_squared(y_true, y_pred)
    expected = 0.0

    assert result == expected


def test_case_5():
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([5, 4, 3, 2, 1])

    result = r_squared(y_true, y_pred)
    expected = -3.0

    assert result == expected
