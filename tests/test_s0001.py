from solutions.s0001_matrix_times_vector import matrix_dot_vector


def test_case_1():
    assert matrix_dot_vector([[1, 2], [2, 4]], [1, 2]) == [5, 10]


def test_case_2():
    assert matrix_dot_vector([[1, 2], [2, 4], [6, 8], [12, 4]], [1, 2, 3]) == -1
