from src.s0065_implement_compressed_row_sparse_matrix_format_conversion import (
    compressed_row_sparse_matrix,
)


def test_case_1():
    dense_matrix = [[1, 0, 0, 0], [0, 2, 0, 0], [3, 0, 4, 0], [1, 0, 0, 5]]

    result = compressed_row_sparse_matrix(dense_matrix)
    expected = (
        [1, 2, 3, 4, 1, 5],
        [0, 1, 0, 2, 0, 3],
        [0, 1, 2, 4, 6],
    )

    assert result[0] == expected[0]
    assert result[1] == expected[1]
    assert result[2] == expected[2]


def test_case_2():
    dense_matrix = [[0, 0, 0], [1, 2, 0], [0, 3, 4]]

    result = compressed_row_sparse_matrix(dense_matrix)
    expected = (
        [1, 2, 3, 4],
        [0, 1, 1, 2],
        [0, 0, 2, 4],
    )

    assert result[0] == expected[0]
    assert result[1] == expected[1]
    assert result[2] == expected[2]


def test_case_3():
    dense_matrix = [
        [0, 0, 3, 0, 0],
        [0, 4, 0, 0, 0],
        [5, 0, 0, 6, 0],
        [0, 0, 0, 0, 0],
        [0, 7, 0, 0, 8],
    ]

    result = compressed_row_sparse_matrix(dense_matrix)
    expected = (
        [3, 4, 5, 6, 7, 8],
        [2, 1, 0, 3, 1, 4],
        [0, 1, 2, 4, 4, 6],
    )

    assert result[0] == expected[0]
    assert result[1] == expected[1]
    assert result[2] == expected[2]
