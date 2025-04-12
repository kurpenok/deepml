import numpy as np

from solutions.s0114_implement_global_average_pooling import global_avg_pool


def test_case_1():
    assert np.allclose(
        global_avg_pool(np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])),
        np.array([5.5, 6.5, 7.5]),
        atol=1e-4,
    )


def test_case_2():
    assert np.allclose(np.array([[[100, 200]]]), np.array([100.0, 200.0]), atol=1e-4)


def test_case_3():
    assert np.allclose(np.ones((3, 3, 1)), np.array([1.0]), atol=1e-4)
