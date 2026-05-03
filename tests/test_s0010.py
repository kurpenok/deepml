from src.s0010_calculate_covariance_matrix import calculate_covariance_matrix


def test_case_1():
    assert calculate_covariance_matrix([[1, 2, 3], [4, 5, 6]]) == [
        [1.0, 1.0],
        [1.0, 1.0],
    ]


def test_case_2():
    assert calculate_covariance_matrix([[1, 4], [2, 5], [3, 6]]) == [
        [4.5, 4.5, 4.5],
        [4.5, 4.5, 4.5],
        [4.5, 4.5, 4.5],
    ]


def test_case_3():
    assert calculate_covariance_matrix([[1, 2, 3]]) == [[1.0]]
