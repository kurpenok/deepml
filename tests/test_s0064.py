from solutions.s0064_implement_gini_impurity_calculation_for_a_set_of_classes import (
    gini_impurity,
)


def test_case_1():
    assert gini_impurity([0, 1, 1, 1, 0]) == 0.48


def test_case_2():
    assert gini_impurity([0, 0, 0, 0, 1, 1, 1, 1]) == 0.5


def test_case_3():
    assert gini_impurity([0, 0, 0, 0, 0, 1]) == 0.278
