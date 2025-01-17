from solutions.s0027_transformation_matrix_from_basis_b_to_c import transform_basis


def test_case_1():
    assert transform_basis(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        [[1, 2.3, 3], [4.4, 25, 6], [7.4, 8, 9]],
    ) == [
        [-0.6772, -0.0126, 0.2342],
        [-0.0184, 0.0505, -0.0275],
        [0.5732, -0.0345, -0.0569],
    ]


def test_case_2():
    assert transform_basis([[1, 0], [0, 1]], [[1, 2], [9, 2]]) == [
        [-0.125, 0.125],
        [0.5625, -0.0625],
    ]
