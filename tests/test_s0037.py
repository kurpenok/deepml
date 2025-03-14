import numpy as np

from solutions.s0037_calculate_correlation_matrix import calculate_correlation_matrix


def test_case_1():
    assert np.allclose(
        calculate_correlation_matrix(np.array([[1, 2], [3, 4], [5, 6]])),
        np.array([[1, 1], [1, 1]]),
    )


def test_case_2():
    assert np.allclose(
        calculate_correlation_matrix(np.array([[1, 2, 3], [7, 15, 6], [7, 8, 9]])),
        np.array(
            [
                [1.0, 0.84298868, 0.8660254],
                [0.84298868, 1, 0.46108397],
                [0.8660254, 0.46108397, 1.0],
            ]
        ),
    )
