from solutions.s0091_calculate_f1_score_from_predicted_and_true_labels import (
    calculate_f1_score,
)


def test_case_1():
    assert calculate_f1_score([1, 0, 1, 1, 0], [1, 0, 0, 1, 1]) == 0.667


def test_case_2():
    assert calculate_f1_score([1, 1, 0, 0], [1, 0, 0, 1]) == 0.5
