from src.s0002_transpose_of_a_matrix import transpose_matrix


def test_case_1():
    assert transpose_matrix([[1, 2, 3], [4, 5, 6]]) == [[1, 4], [2, 5], [3, 6]]


def test_case_2():
    assert transpose_matrix([[1, 2, 3, 4]]) == [[1], [2], [3], [4]]


def test_case_3():
    assert transpose_matrix([]) == []


def test_case_4():
    assert transpose_matrix([[42]]) == [[42]]
