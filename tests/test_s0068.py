import numpy as np

from solutions.s0068_find_the_image_of_a_matrix_using_row_echelon_form import (
    matrix_image,
)


def test_case_1():
    assert (
        matrix_image(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
        == np.array([[1, 2], [4, 5], [7, 8]])
    ).all()


def test_case_2():
    assert (
        matrix_image(np.array([[1, 0], [0, 1]])) == np.array([[1, 0], [0, 1]])
    ).all()


def test_case_3():
    assert (matrix_image(np.array([[1, 2], [2, 4]])) == np.array([[1], [2]])).all()
