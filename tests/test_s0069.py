import numpy as np

from solutions.s0069_calculate_r_squared_for_regression_analysis import r_squared


def test_case_1():
    assert (
        r_squared(np.array([1, 2, 3, 4, 5]), np.array([1.1, 2.1, 2.9, 4.2, 4.8]))
        == 0.989
    )


def test_case_2():
    assert r_squared(np.array([1, 2, 3, 4, 5]), np.array([1, 2, 3, 4, 5])) == 1
