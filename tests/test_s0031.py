import numpy as np

from solutions.s0031_divide_dataset_based_on_feature_threshold import divide_on_feature


def test_case_1():
    result = divide_on_feature(
        np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]), 0, 5
    )
    expected_result = [np.array([[5, 6], [7, 8], [9, 10]]), np.array([[1, 2], [3, 4]])]

    for i in range(len(result)):
        assert np.all(result[i] == expected_result[i])


def test_case_2():
    result = divide_on_feature(np.array([[1, 1], [2, 2], [3, 3], [4, 4]]), 1, 3)
    expected_result = [np.array([[3, 3], [4, 4]]), np.array([[1, 1], [2, 2]])]

    for i in range(len(result)):
        assert np.all(result[i] == expected_result[i])
