import numpy as np

from solutions.s0061_implement_f_score_calculation_for_binary_classification import (
    f_score,
)


def test_case_1():
    assert (
        f_score(np.array([1, 0, 1, 1, 0, 1]), np.array([1, 0, 1, 0, 0, 1]), 1) == 0.857
    )


def test_case_2():
    assert f_score(np.array([1, 0, 1, 1, 0, 0]), np.array([1, 0, 0, 0, 0, 1]), 1) == 0.4
