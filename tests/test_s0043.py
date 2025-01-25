import numpy as np

from solutions.s0043_implement_ridge_regression_loss_function import ridge_loss


def test_case_1():
    assert (
        ridge_loss(
            np.array([[1, 1], [2, 1], [3, 1], [4, 1]]),
            np.array([0.2, 2]),
            np.array([2, 3, 4, 5]),
            0.1,
        )
        == 2.204
    )


def test_case_2():
    assert (
        ridge_loss(
            np.array([[1, 1, 4], [2, 1, 2], [3, 1, 0.1], [4, 1, 1.2], [1, 2, 3]]),
            np.array([0.2, 2, 5]),
            np.array([2, 3, 4, 5, 2]),
            0.1,
        )
        == 164.402
    )
