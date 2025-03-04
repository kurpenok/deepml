from solutions.s0008_calculate_2x2_matrix_inverse import inverse_2x2


def test_case_1():
    assert inverse_2x2([[4, 7], [2, 6]]) == [[0.6, -0.7], [-0.2, 0.4]]


def test_case_2():
    assert inverse_2x2([[2, 1], [6, 2]]) == [[-1.0, 0.5], [3.0, -1.0]]
