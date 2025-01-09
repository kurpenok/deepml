from solutions.s0005_scalar_multiplication_of_a_matrix import scalar_multiply


def test_case_1():
    assert scalar_multiply([[1, 2], [3, 4]], 2) == [[2, 4], [6, 8]]


def test_case_2():
    assert scalar_multiply([[0, -1], [1, 0]], -1) == [[0, 1], [-1, 0]]
