from solutions.s0065_implement_compressed_row_sparse_matrix_csr_format_conversion import (
    compressed_row_sparse_matrix,
)


def test_case_1():
    assert all(
        (a == b)
        for a, b in zip(
            compressed_row_sparse_matrix(
                [[1, 0, 0, 0], [0, 2, 0, 0], [3, 0, 4, 0], [1, 0, 0, 5]]
            ),
            (
                [1, 2, 3, 4, 1, 5],
                [0, 1, 0, 2, 0, 3],
                [0, 1, 2, 4, 6],
            ),
        )
    )


def test_case_2():
    assert all(
        (a == b)
        for a, b in zip(
            compressed_row_sparse_matrix([[0, 0, 0], [1, 2, 0], [0, 3, 4]]),
            (
                [1, 2, 3, 4],
                [0, 1, 1, 2],
                [0, 0, 2, 4],
            ),
        )
    )
