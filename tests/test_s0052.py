import numpy as np

from solutions.s0052_implement_recall_metric_in_binary_classification import recall


def test_case_1():
    assert recall(np.array([1, 0, 1, 1, 0, 1]), np.array([1, 0, 1, 0, 0, 1])) == 0.75


def test_case_2():
    assert recall(np.array([1, 0, 1, 1, 0, 0]), np.array([1, 0, 0, 0, 0, 1])) == 0.333
