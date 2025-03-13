import numpy as np

from solutions.s0033_generate_random_subsets_of_a_dataset import get_random_subsets


def test_case_1():
    assert get_random_subsets(
        np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]),
        np.array([1, 2, 3, 4, 5]),
        3,
        False,
    ) == [
        ([[3, 4], [9, 10]], [2, 5]),
        ([[7, 8], [3, 4]], [4, 2]),
        ([[3, 4], [1, 2]], [2, 1]),
    ]


def test_case_2():
    assert get_random_subsets(
        np.array([[1, 1], [2, 2], [3, 3], [4, 4]]),
        np.array([10, 20, 30, 40]),
        1,
        True,
        42,
    ) == [([[3, 3], [4, 4], [1, 1], [3, 3]], [30, 40, 10, 30])]
