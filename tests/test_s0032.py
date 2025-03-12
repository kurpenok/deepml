import numpy as np

from solutions.s0032_generate_polynomial_features import polynomial_features


def test_case_1():
    assert (
        polynomial_features(np.array([[2, 3], [3, 4], [5, 6]]), 2)
        == np.array(
            [
                [1, 2, 3, 4, 6, 9],
                [1, 3, 4, 9, 12, 16],
                [1, 5, 6, 25, 30, 36],
            ]
        )
    ).all()


def test_case_2():
    assert (
        polynomial_features(np.array([[1, 2], [3, 4], [5, 6]]), 3)
        == np.array(
            [
                [1, 1, 2, 1, 2, 4, 1, 2, 4, 8],
                [1, 3, 4, 9, 12, 16, 27, 36, 48, 64],
                [1, 5, 6, 25, 30, 36, 125, 150, 180, 216],
            ]
        )
    ).all()
