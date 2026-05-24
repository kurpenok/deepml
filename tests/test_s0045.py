import numpy as np

from src.s0045_linear_kernel_function import kernel_function


def test_case_1():
    x1 = np.array([1, 2, 3])
    x2 = np.array([4, 5, 6])

    result = kernel_function(x1, x2)
    expected = 32

    assert result == expected


def test_case_2():
    x1 = np.array([0, 1, 2])
    x2 = np.array([3, 4, 5])

    result = kernel_function(x1, x2)
    expected = 14

    assert result == expected
