from solutions.s0004_calculate_mean_by_row_or_column import calculate_matrix_mean


def test_case_1():
    assert calculate_matrix_mean([[1, 2, 3], [4, 5, 6], [7, 8, 9]], "column") == [
        4.0,
        5.0,
        6.0,
    ]


def test_case_2():
    assert calculate_matrix_mean([[1, 2, 3], [4, 5, 6], [7, 8, 9]], "row") == [
        2.0,
        5.0,
        8.0,
    ]
