import numpy as np

from solutions.s0083_dot_product_calculator import calculate_dot_product


def test_case_1():
    assert calculate_dot_product(np.array([1, 2, 3]), np.array([4, 5, 6])) == 32


def test_case_2():
    assert calculate_dot_product(np.array([-1, 2, 3]), np.array([4, -5, 6])) == 4
