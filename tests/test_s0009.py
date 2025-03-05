from solutions.s0009_matrix_times_matrix import matrixmul


def test_case_1():
    assert matrixmul([[1, 2], [2, 4]], [[2, 1], [3, 4]]) == [[8, 9], [16, 18]]


def test_case_2():
    assert matrixmul(
        [[1, 2, 3], [2, 3, 4], [5, 6, 7]], [[3, 2, 1], [4, 3, 2], [5, 4, 3]]
    ) == [[26, 20, 14], [38, 29, 20], [74, 56, 38]]


def test_case_3():
    assert matrixmul([[0, 0], [2, 4], [1, 2]], [[0, 0], [2, 4]]) == [
        [0, 0],
        [8, 16],
        [4, 8],
    ]
