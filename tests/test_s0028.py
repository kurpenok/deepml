import numpy as np

from solutions.s0028_svd_a_2x2_matrix_using_eigen_values_and_vectors import svd_2x2


def test_case_1():
    result = svd_2x2(np.array([[-10, 8], [10, -1]]))

    assert np.allclose(result[0], np.array([[0.8, -0.6], [-0.6, -0.8]]))
    assert np.allclose(result[1], np.array([15.65247584, 4.47213595]))
    assert np.allclose(
        result[2], np.array([[-0.89442719, 0.4472136], [-0.4472136, -0.89442719]])
    )
