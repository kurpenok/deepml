from solutions.s0119_solve_system_of_linear_equations_using_cramers_rule import (
    cramers_rule,
)


def test_case_1():
    assert cramers_rule([[2, -1, 3], [4, 2, 1], [-6, 1, -2]], [5, 10, -3]) == [
        0.1667,
        3.3333,
        2.6667,
    ]


def test_case_2():
    assert cramers_rule([[1, 2], [3, 4]], [5, 6]) == [-4.0, 4.5]


def test_case_3():
    assert cramers_rule([[1, 2], [2, 4]], [3, 6]) == -1
