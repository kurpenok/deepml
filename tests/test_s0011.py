import numpy as np

from solutions.s0011_solve_linear_equations_using_jacobi_method import solve_jacobi


def test_case_1():
    assert solve_jacobi(
        np.array([[5, -2, 3], [-3, 9, 1], [2, -1, -7]]), np.array([-1, 2, 3]), 2
    ) == [0.146, 0.2032, -0.5175]


def test_case_2():
    assert solve_jacobi(
        np.array([[4, 1, 2], [1, 5, 1], [2, 1, 3]]), np.array([4, 6, 7]), 5
    ) == [-0.0806, 0.9324, 2.4422]
