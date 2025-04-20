import numpy as np

from solutions.s0012_singular_value_decomposition import svd_2x2_singular_values


def test_case_1():
    result = svd_2x2_singular_values(np.array([[2, 1], [1, 2]]))

    assert np.allclose(
        result[0],
        np.array([[0.70710678, -0.70710678], [0.70710678, 0.70710678]]),
        atol=1e-4,
    )
    assert np.allclose(result[1], np.array([3.0, 1.0]), atol=1e-4)
    assert np.allclose(
        result[2],
        np.array([[0.70710678, 0.70710678], [-0.70710678, 0.70710678]]),
        atol=1e-4,
    )


def test_case_2():
    result = svd_2x2_singular_values(np.array([[1, 2], [3, 4]]))

    assert np.allclose(
        result[0],
        np.array([[0.40455358, 0.9145143], [0.9145143, -0.40455358]]),
        atol=1e-4,
    )
    assert np.allclose(result[1], np.array([5.4649857, 0.36596619]), atol=1e-4)
    assert np.allclose(
        result[2],
        np.array([[0.57604844, 0.81741556], [-0.81741556, 0.57604844]]),
        atol=1e-4,
    )
