import numpy as np

from solutions.s0018_implement_k_fold_cross_validation import k_fold_cross_validation


def test_case_1():
    assert k_fold_cross_validation(
        np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
        np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
        k=5,
        shuffle=False,
    ) == [
        ([2, 3, 4, 5, 6, 7, 8, 9], [0, 1]),
        ([0, 1, 4, 5, 6, 7, 8, 9], [2, 3]),
        ([0, 1, 2, 3, 6, 7, 8, 9], [4, 5]),
        ([0, 1, 2, 3, 4, 5, 8, 9], [6, 7]),
        ([0, 1, 2, 3, 4, 5, 6, 7], [8, 9]),
    ]


def test_case_2():
    assert k_fold_cross_validation(
        np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
        np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
        k=2,
        shuffle=True,
        random_seed=42,
    ) == [([2, 9, 4, 3, 6], [8, 1, 5, 0, 7]), ([8, 1, 5, 0, 7], [2, 9, 4, 3, 6])]
