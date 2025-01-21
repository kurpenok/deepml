import numpy as np

from solutions.s0035_convert_vector_to_diagonal_matrix import make_diagonal


def test_case_1():
    assert (
        make_diagonal(np.array([1, 2, 3]))
        == np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
    ).all()


def test_case_2():
    assert (
        make_diagonal(np.array([4, 5, 6, 7]))
        == np.array([[4, 0, 0, 0], [0, 5, 0, 0], [0, 0, 6, 0], [0, 0, 0, 7]])
    ).all()
