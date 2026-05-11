import numpy as np

from src.s0034_one_hot_encoding_of_nominal_values import to_categorical


def test_case_1():
    X = np.array([0, 1, 2, 1, 0])

    result = to_categorical(X)
    expected = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )

    assert np.all(result == expected)


def test_case_2():
    X = np.array([3, 1, 2, 1, 3])
    n_col = 4

    result = to_categorical(X, n_col)
    expected = np.array(
        [
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    assert np.all(result == expected)
