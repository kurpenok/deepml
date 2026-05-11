import numpy as np

from src.s0035_convert_vector_to_diagonal_matrix import make_diagonal


def test_case_1():
    X = np.array([1, 2, 3])

    result = make_diagonal(X)
    expected = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]])

    assert np.all(result == expected)


def test_case_2():
    X = np.array([4, 5, 6, 7])

    result = make_diagonal(X)
    expected = np.array([[4, 0, 0, 0], [0, 5, 0, 0], [0, 0, 6, 0], [0, 0, 0, 7]])

    assert np.all(result == expected)
