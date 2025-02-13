from solutions.s0081_poisson_distribution_probability_calculator import (
    poisson_probability,
)


def test_case_1():
    assert poisson_probability(3, 5) == 0.14037


def test_case_2():
    assert poisson_probability(0, 5) == 0.00674
