import numpy as np

from src.s0027_transformation_matrix_from_basis_b_to_c import transform_basis


def test_case_1():
    B = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    C = [[1.0, 2.3, 3.0], [4.4, 25.0, 6.0], [7.4, 8.0, 9.0]]

    result = transform_basis(B, C)
    expected = [
        [-0.6772, -0.0126, 0.2342],
        [-0.0184, 0.0505, -0.0275],
        [0.5732, -0.0345, -0.0569],
    ]

    assert np.allclose(result, expected, atol=1e-4)


def test_case_2():
    B = [[1.0, 0.0], [0.0, 1.0]]
    C = [[1.0, 2.0], [9.0, 2.0]]

    result = transform_basis(B, C)
    expected = [[-0.125, 0.125], [0.5625, -0.0625]]

    assert np.allclose(result, expected, atol=1e-4)
