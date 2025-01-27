from solutions.s0045_linear_kernel_function import kernel_function


def test_case_1():
    assert kernel_function([1, 2, 3], [4, 5, 6]) == 32


def test_case_2():
    assert kernel_function([0, 1, 2], [3, 4, 5]) == 14
