from solutions.s0013_determinant_of_a_4x4_matrix_using_laplace_expansion import (
    determinant_4x4,
)


def test_case_1():
    assert (
        determinant_4x4([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
        == 0
    )


def test_case_2():
    assert (
        determinant_4x4([[4, 3, 2, 1], [3, 2, 1, 4], [2, 1, 4, 3], [1, 4, 3, 2]])
        == -160
    )


def test_case_3():
    assert (
        determinant_4x4([[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]]) == 0
    )
