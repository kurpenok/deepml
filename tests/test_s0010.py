from solutions.s0010_calculate_covariance_matrix import calculate_covariance_matrix


def test_case_1():
    assert calculate_covariance_matrix([[1, 2, 3], [4, 5, 6]]) == [
        [1.0, 1.0],
        [1.0, 1.0],
    ]


def test_case_2():
    assert calculate_covariance_matrix([[1, 5, 6], [2, 3, 4], [7, 8, 9]]) == [
        [7.0, 2.5, 2.5],
        [2.5, 1.0, 1.0],
        [2.5, 1.0, 1.0],
    ]
