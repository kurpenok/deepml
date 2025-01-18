import numpy as np

from solutions.s0029_random_shuffle_of_dataset import shuffle_data


def test_case_1():
    result = shuffle_data(
        np.array([[1, 2], [3, 4], [5, 6], [7, 8]]),
        np.array([1, 2, 3, 4]),
        seed=42,
    )

    expected_result = (
        np.array([[3, 4], [7, 8], [1, 2], [5, 6]]),
        np.array([2, 4, 1, 3]),
    )

    assert np.all(result[0] == expected_result[0])
    assert np.all(result[1] == expected_result[1])


def test_case_2():
    result = shuffle_data(
        np.array([[1, 1], [2, 2], [3, 3], [4, 4]]),
        np.array([10, 20, 30, 40]),
        seed=24,
    )

    expected_result = (
        np.array([[4, 4], [2, 2], [1, 1], [3, 3]]),
        np.array([40, 20, 10, 30]),
    )

    assert np.all(result[0] == expected_result[0])
    assert np.all(result[1] == expected_result[1])
