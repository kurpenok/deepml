import numpy as np

from solutions.s0015_linear_regression_using_gradient_descent import (
    linear_regression_gradient_descent,
)


def test_case_1():
    assert (
        linear_regression_gradient_descent(
            np.array([[1, 1], [1, 2], [1, 3]]), np.array([1, 2, 3]), 0.01, 1000
        )
        == np.array([0.1107, 0.9513])
    ).all()
