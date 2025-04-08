import numpy as np

from solutions.s0117_compute_orthonormal_basis_for_2d_vectors import orthonormal_basis


def test_case_1():
    result = orthonormal_basis([[1.0, 0.0], [1.0, 1.0]])

    assert np.allclose(result[0], np.array([1.0, 0.0]), atol=1e-4)
    assert np.allclose(result[1], np.array([0.0, 1.0]), atol=1e-4)


def test_case_2():
    result = orthonormal_basis([[2, 0], [4, 0]], tol=1e-10)

    assert np.allclose(result[0], np.array([1.0, 0.0]), atol=1e-4)


def test_case_3():
    result = orthonormal_basis([[1, 1], [1, -1]], tol=1e-5)

    assert np.allclose(result[0], np.array([0.7071, 0.7071]), atol=1e-4)
    assert np.allclose(result[1], np.array([0.7071, -0.7071]), atol=1e-4)


def test_case_4():
    assert orthonormal_basis([[0, 0]], tol=1e-10) == []
