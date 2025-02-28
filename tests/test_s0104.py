import numpy as np

from solutions.s0104_binary_classification_with_logistic_regression import (
    predict_logistic,
)


def test_case_1():
    assert (
        predict_logistic(
            np.array([[1, 1], [2, 2], [-1, -1], [-2, -2]]), np.array([1, 1]), 0
        )
        == np.array([1, 1, 0, 0])
    ).all()


def test_case_2():
    assert (
        predict_logistic(
            np.array([[0, 0], [0.1, 0.1], [-0.1, -0.1]]), np.array([1, 1]), 0
        )
        == np.array([1, 1, 0])
    ).all()
