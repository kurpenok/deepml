import numpy as np

from solutions.s0118_compute_the_cross_product_of_two_3d_vectors import cross_product


def test_case_1():
    assert np.allclose(
        cross_product(np.array([1, 0, 0]), np.array([0, 1, 0])),
        np.array([0, 0, 1]),
        atol=1e-4,
    )


def test_case_2():
    assert np.allclose(
        cross_product(np.array([0, 1, 0]), np.array([0, 0, 1])),
        np.array([1, 0, 0]),
        atol=1e-4,
    )


def test_case_3():
    assert np.allclose(
        cross_product(np.array([1, 2, 3]), np.array([4, 5, 6])),
        np.array([-3, 6, -3]),
        atol=1e-4,
    )


def test_case_4():
    assert np.allclose(
        cross_product(np.array([1, 0, 0]), np.array([1, 0, 0])),
        np.array([0, 0, 0]),
        atol=1e-4,
    )
