from solutions.s0067_implement_compressed_column_sparse_matrix_format import (
    compressed_col_sparse_matrix,
)


def test_case_1():
    assert compressed_col_sparse_matrix([[0, 0, 3, 0], [1, 0, 0, 4], [0, 2, 0, 0]]) == (
        [1, 2, 3, 4],
        [1, 2, 0, 1],
        [0, 1, 2, 3, 4],
    )


def test_case_2():
    assert compressed_col_sparse_matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]]) == (
        [],
        [],
        [0, 0, 0, 0],
    )


def test_case_3():
    assert compressed_col_sparse_matrix([[0, 0, 0], [1, 2, 0], [0, 3, 4]]) == (
        [1, 2, 3, 4],
        [1, 1, 2, 2],
        [0, 1, 3, 4],
    )
