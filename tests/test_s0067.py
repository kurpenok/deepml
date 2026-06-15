from src.s0067_implement_compressed_column_sparse_matrix_format import (
    compressed_col_sparse_matrix,
)


def test_case_1():
    dense_matrix = [[0, 0, 3, 0], [1, 0, 0, 4], [0, 2, 0, 0]]

    result = compressed_col_sparse_matrix(dense_matrix)
    expected = ([1, 2, 3, 4], [1, 2, 0, 1], [0, 1, 2, 3, 4])

    assert result[0] == expected[0]
    assert result[1] == expected[1]
    assert result[2] == expected[2]


def test_case_2():
    dense_matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

    result = compressed_col_sparse_matrix(dense_matrix)
    expected = ([], [], [0, 0, 0, 0])

    assert result[0] == expected[0]
    assert result[1] == expected[1]
    assert result[2] == expected[2]


def test_case_3():
    dense_matrix = [[0, 0, 0], [1, 2, 0], [0, 3, 4]]

    result = compressed_col_sparse_matrix(dense_matrix)
    expected = ([1, 2, 3, 4], [1, 1, 2, 2], [0, 1, 3, 4])

    assert result[0] == expected[0]
    assert result[1] == expected[1]
    assert result[2] == expected[2]
