from solutions.s0002_transpose_of_a_matrix import transpose_matrix


def test_case_1():
    assert transpose_matrix([[1, 2], [3, 4], [5, 6]]) == [[1, 3, 5], [2, 4, 6]]


def test_case_2():
    assert transpose_matrix([[1, 2, 3], [4, 5, 6]]) == [[1, 4], [2, 5], [3, 6]]
