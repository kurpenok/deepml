import numpy as np

from solutions.s0073_calculate_dice_score_for_classification import dice_score


def test_case_1():
    assert (
        dice_score(np.array([1, 1, 0, 1, 0, 1]), np.array([1, 1, 0, 0, 0, 1])) == 0.857
    )


def test_case_2():
    assert dice_score(np.array([1, 1, 0, 0]), np.array([1, 1, 0, 0])) == 1


def test_case_3():
    assert dice_score(np.array([1, 1, 0, 0]), np.array([0, 0, 1, 1])) == 0
