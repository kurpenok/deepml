import numpy as np

from solutions.s0046_implement_precision_metric import precision


def test_case_1():
    assert precision(np.array([1, 0, 1, 1, 0, 1]), np.array([1, 0, 1, 0, 0, 1])) == 1


def test_case_2():
    assert precision(np.array([1, 0, 1, 1, 0, 0]), np.array([1, 0, 0, 0, 0, 1])) == 0.5
