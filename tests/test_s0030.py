import numpy as np

from src.s0030_batch_iterator_for_dataset import batch_iterator


def test_case_1():
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    y = np.array([1, 2, 3, 4, 5])
    batch_size = 2

    result = batch_iterator(X, y, batch_size)
    expected = [
        [[[1, 2], [3, 4]], [1, 2]],
        [[[5, 6], [7, 8]], [3, 4]],
        [[[9, 10]], [5]],
    ]

    for i in range(len(expected)):
        assert np.all(result[i][0] == expected[i][0])
        assert np.all(result[i][1] == expected[i][1])


def test_case_2():
    X = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])
    batch_size = 3

    result = batch_iterator(X, batch_size=batch_size)
    expected = [[[1, 1], [2, 2], [3, 3]], [[4, 4]]]

    for i in range(len(expected)):
        assert np.all(result[i][0] == expected[i][0])
