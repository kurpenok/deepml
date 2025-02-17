from solutions.s0086_detect_overfitting_or_underfitting import model_fit_quality


def test_case_1():
    assert model_fit_quality(0.95, 0.65) == 1


def test_case_2():
    assert model_fit_quality(0.6, 0.5) == -1
