import numpy as np

from solutions.s0050_implement_lasso_regression_using_gradient_descent import (
    l1_regularization_gradient_descent,
)


def test_case_1():
    weights, bias = l1_regularization_gradient_descent(
        np.array([[0, 0], [1, 1], [2, 2]]), np.array([0, 1, 2])
    )

    assert np.allclose(weights, np.array([0.425, 0.425]), atol=0.01)
    assert np.isclose(bias, 0.15, atol=0.01)


def test_case_2():
    weights, bias = l1_regularization_gradient_descent(
        np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]]), np.array([1, 2, 3, 4, 5])
    )

    assert np.allclose(weights, np.array([0.273, 0.681]), atol=0.01)
    assert np.isclose(bias, 0.41, atol=0.01)
