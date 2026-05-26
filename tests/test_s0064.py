import numpy as np

from src.s0064_implement_gini_impurity_calculation_for_a_set_of_classes import (
    gini_impurity,
)


def test_case_1():
    y = np.array([0, 1, 1, 1, 0])

    result = gini_impurity(y)
    expected = 0.48

    assert result == expected


def test_case_2():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    result = gini_impurity(y)
    expected = 0.5

    assert result == expected


def test_case_3():
    y = np.array([0, 0, 0, 0, 0, 1])

    result = gini_impurity(y)
    expected = 0.2777777777777778

    assert result == expected


def test_case_4():
    y = np.array([0, 1, 2, 2, 2, 1, 2])

    result = gini_impurity(y)
    expected = 0.5714285714285714

    assert result == expected
