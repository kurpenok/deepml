import numpy as np

from src.s0052_implement_recall_metric_in_binary_classification import recall


def test_case_1():
    y_true = np.array([1, 0, 1, 1, 0, 1])
    y_pred = np.array([1, 0, 1, 0, 0, 1])

    result = recall(y_true, y_pred)
    expected = 0.75

    assert result == expected


def test_case_2():
    y_true = np.array([1, 0, 1, 1, 0, 0])
    y_pred = np.array([1, 0, 0, 0, 0, 1])

    result = recall(y_true, y_pred)
    expected = 1 / 3

    assert result == expected


def test_case_3():
    y_true = np.array([1, 0, 1, 1, 0, 0])
    y_pred = np.array([1, 0, 1, 1, 0, 0])

    result = recall(y_true, y_pred)
    expected = 1.0

    assert result == expected


def test_case_4():
    y_true = np.array([1, 0, 1, 1, 0, 1])
    y_pred = np.array([0, 0, 0, 1, 0, 1])

    result = recall(y_true, y_pred)
    expected = 0.5

    assert result == expected


def test_case_5():
    y_true = np.array([1, 0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1, 0])

    result = recall(y_true, y_pred)
    expected = 0.0

    assert result == expected
