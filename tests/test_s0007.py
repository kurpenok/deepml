from solutions.s0007_matrix_transformation import transform_matrix


def test_case_1():
    assert transform_matrix([[1, 2], [3, 4]], [[2, 0], [0, 2]], [[1, 1], [0, 1]]) == [
        [0.5, 1.5],
        [1.5, 3.5],
    ]


def test_case_2():
    assert transform_matrix([[1, 0], [0, 1]], [[1, 2], [3, 4]], [[2, 0], [0, 2]]) == [
        [-3.999999999999999, 1.9999999999999996],
        [2.9999999999999996, -0.9999999999999998],
    ]
