import numpy as np

from solutions.s0030_batch_iterator_for_dataset import batch_iterator


def test_case_1():
    result = batch_iterator(
        np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]),
        np.array([1, 2, 3, 4, 5]),
        2,
    )

    expected_result = [
        [[[1, 2], [3, 4]], [1, 2]],
        [[[5, 6], [7, 8]], [3, 4]],
        [[[9, 10]], [5]],
    ]

    for i in range(len(expected_result)):
        assert np.all(result[i][0] == expected_result[i][0])
        assert np.all(result[i][1] == expected_result[i][1])


def test_case_2():
    result = batch_iterator(np.array([[1, 1], [2, 2], [3, 3], [4, 4]]), batch_size=3)

    expected_result = [[[1, 1], [2, 2], [3, 3]], [[4, 4]]]

    for i in range(len(expected_result)):
        assert np.all(result[i][0] == expected_result[i][0])
