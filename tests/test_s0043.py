import numpy as np

from src.s0043_implement_ridge_regression_loss_function import ridge_loss


def test_case_1():
    X = np.array([[1, 1], [2, 1], [3, 1], [4, 1]])
    w = np.array([0.2, 2])
    y_true = np.array([2, 3, 4, 5])
    alpha = 0.1

    result = ridge_loss(X, w, y_true, alpha)
    expected = 2.204

    assert result == expected


def test_case_2():
    X = np.array([[1, 1, 4], [2, 1, 2], [3, 1, 0.1], [4, 1, 1.2], [1, 2, 3]])
    w = np.array([0.2, 2, 5])
    y_true = np.array([2, 3, 4, 5, 2])
    alpha = 0.1

    result = ridge_loss(X, w, y_true, alpha)
    expected = 164.402

    assert result == expected
