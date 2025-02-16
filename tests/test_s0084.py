from solutions.s0084_phi_transformation_for_polynomial_features import phi_transform


def test_case_1():
    assert phi_transform([1.0, 2.0], 2) == [[1.0, 1.0, 1.0], [1.0, 2.0, 4.0]]


def test_case_2():
    assert phi_transform([], 2) == []


def test_case_3():
    assert phi_transform([1.0, 2.0], -1) == [[], []]
