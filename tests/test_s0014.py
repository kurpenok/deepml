from solutions.s0014_linear_regression_using_normal_equation import (
    linear_regression_normal_equation,
)


def test_case_1():
    assert linear_regression_normal_equation([[1, 1], [1, 2], [1, 3]], [1, 2, 3]) == [
        0.0,
        1.0,
    ]


def test_case_2():
    assert linear_regression_normal_equation(
        [[1, 3, 4], [1, 2, 5], [1, 3, 2]], [1, 2, 1]
    ) == [4.0, -1.0, -0.0]
