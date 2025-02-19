import numpy as np

from solutions.s0093_calculate_mean_absolute_error import mae


def test_case_1():
    assert mae(np.array([3, -0.5, 2, 7]), np.array([2.5, 0.0, 2, 8])) == 0.5


def test_case_2():
    assert (
        mae(
            np.array([[0.5, 1], [-1, 1], [7, -6]]), np.array([[0, 2], [-1, 2], [8, -5]])
        )
        == 0.75
    )
