import numpy as np

from solutions.s0048_implement_reduced_row_echelon_form_function import rref


def test_case_1():
    assert (
        rref(np.array([[1, 2, -1, -4], [2, 3, -1, -11], [-2, 0, -3, 22]]))
        == np.array([[1, 0, 0, -8], [0, 1, 0, 1], [-0, -0, 1, -2]])
    ).all()


def test_case_2():
    assert (
        rref(np.array([[2, 4, -2], [4, 9, -3], [-2, -3, 7]]))
        == np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    ).all()
