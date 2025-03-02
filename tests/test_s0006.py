from solutions.s0006_calculate_eigenvalues_of_a_matrix import calculate_eigenvalues


def test_case_1():
    assert calculate_eigenvalues([[2, 1], [1, 2]]) == [3.0, 1.0]


def test_case_2():
    assert calculate_eigenvalues([[4, -2], [1, 1]]) == [3.0, 2.0]
