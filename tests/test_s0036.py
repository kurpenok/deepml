import numpy as np

from solutions.s0036_calculate_accuracy_score import accuracy_score


def test_case_1():
    assert (
        accuracy_score(np.array([1, 0, 1, 1, 0, 1]), np.array([1, 0, 0, 1, 0, 1]))
        == 0.8333333333333334
    )


def test_case_2():
    assert accuracy_score(np.array([1, 1, 1, 1]), np.array([1, 0, 1, 0])) == 0.5
