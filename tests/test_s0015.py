import numpy as np

from src.s0015_linear_regression_using_gradient_descent import (
    linear_regression_gradient_descent,
)


def test_case_1():
    X = np.array([[1, 1], [1, 2], [1, 3]])
    y = np.array([3, 5, 7])

    result = linear_regression_gradient_descent(X, y, alpha=0.1, iterations=1000)
    expected = np.array([1.0, 2.0])

    assert np.allclose(result, expected, atol=1e-4)


def test_case_2():
    X = np.array([[1, 2], [1, 3], [1, 4], [1, 5]])
    y = np.array([7, 9, 11, 13])

    result = linear_regression_gradient_descent(X, y, alpha=0.05, iterations=1000)
    expected = np.array([[2.9699, 2.0078]])

    assert np.allclose(result, expected, atol=1e-4)


def test_case_3():
    X = np.array([[1, 0, 0], [1, 1, 0], [1, 0, 1], [1, 1, 1]])
    y = np.array([1, 3, 4, 6])

    result = linear_regression_gradient_descent(X, y, alpha=0.1, iterations=2000)
    expected = np.array([1.0, 2.0, 3.0])

    assert np.allclose(result, expected, atol=1e-4)
