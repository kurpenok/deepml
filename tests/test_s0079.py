from solutions.s0079_binomial_distribution_probability import binomial_probability


def test_case_1():
    assert binomial_probability(6, 2, 0.5) == 0.23438


def test_case_2():
    assert binomial_probability(6, 4, 0.7) == 0.32414


def test_case_3():
    assert binomial_probability(3, 3, 0.9) == 0.729


def test_case_4():
    assert binomial_probability(5, 0, 0.3) == 0.16807


def test_case_5():
    assert binomial_probability(7, 2, 0.1) == 0.124
