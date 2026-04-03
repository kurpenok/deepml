from src.s0005_scalar_multiplication_of_a_matrix import scalar_multiply


def test_case_1():
    assert scalar_multiply([[1, 2], [3, 4]], 2) == [[2, 4], [6, 8]]


def test_case_2():
    assert scalar_multiply([[1.5, 2.5], [3.5, 4.5]], 2) == [[3.0, 5.0], [7.0, 9.0]]


def test_case_3():
    assert scalar_multiply([[1, 2], [3, 4]], 0) == [[0, 0], [0, 0]]
