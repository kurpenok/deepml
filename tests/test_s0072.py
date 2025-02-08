import numpy as np

from solutions.s0072_calculate_jaccard_index_for_binary_classification import (
    jaccard_index,
)


def test_case_1():
    assert (
        jaccard_index(np.array([1, 0, 1, 1, 0, 1]), np.array([1, 0, 1, 0, 0, 1]))
        == 0.75
    )


def test_case_2():
    assert (
        jaccard_index(np.array([1, 0, 1, 1, 0, 1]), np.array([1, 0, 1, 1, 0, 1])) == 1
    )


def test_case_3():
    assert (
        jaccard_index(np.array([1, 0, 1, 1, 0, 0]), np.array([0, 1, 0, 0, 1, 1])) == 0
    )
