from solutions.s0112_min_max_normalization_of_feature_values import min_max


def test_case_1():
    assert min_max([1, 2, 3, 4, 5]) == [0.0, 0.25, 0.5, 0.75, 1.0]


def test_case_2():
    assert min_max([30, 45, 56, 70, 88]) == [0.0, 0.2586, 0.4483, 0.6897, 1.0]


def test_case_3():
    assert min_max([5, 5, 5, 5]) == [0.0, 0.0, 0.0, 0.0]
