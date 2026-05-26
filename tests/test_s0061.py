import numpy as np

from src.s0061_implement_f_score_calculation_for_binary_classification import f_score


def test_case_1():
    y_true = np.array([1, 0, 1, 1, 0, 1])
    y_pred = np.array([1, 0, 1, 0, 0, 1])
    beta = 1

    result = f_score(y_true, y_pred, beta)
    expected = 0.8571428571428571

    assert result == expected


def test_case_2():
    y_true = np.array([1, 0, 1, 1, 0, 0])
    y_pred = np.array([1, 0, 0, 0, 0, 1])
    beta = 1

    result = f_score(y_true, y_pred, beta)
    expected = 0.4

    assert result == expected


def test_case_3():
    y_true = np.array([1, 0, 1, 1, 0, 1])
    y_pred = np.array([1, 0, 1, 1, 0, 1])
    beta = 2

    result = f_score(y_true, y_pred, beta)
    expected = 1.0

    assert result == expected


def test_case_4():
    y_true = np.array([1, 0, 1, 1, 0, 1])
    y_pred = np.array([0, 0, 0, 1, 0, 1])
    beta = 2

    result = f_score(y_true, y_pred, beta)
    expected = 0.5555555555555556

    assert result == expected
