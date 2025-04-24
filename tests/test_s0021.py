import numpy as np

from solutions.s0021_pegasos_kernel_svm_implementation import pegasos_kernel_svm


def test_case_1():
    assert pegasos_kernel_svm(
        np.array([[1, 2], [2, 3], [3, 1], [4, 1]]),
        np.array([1, 1, -1, -1]),
        "linear",
        0.01,
        100,
    ) == ([100.0, 0.0, -100.0, -100.0], -937.4755)


def test_case_2():
    assert pegasos_kernel_svm(
        np.array([[1, 2], [2, 3], [3, 1], [4, 1]]),
        np.array([1, 1, -1, -1]),
        "rbf",
        0.01,
        100,
        0.5,
    ) == ([100.0, 99.0, -100.0, -100.0], -115.0)
