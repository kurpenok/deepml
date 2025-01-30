from solutions.s0056_kl_divergence_between_two_normal_distributions import (
    kl_divergence_normal,
)


def test_case_1():
    assert kl_divergence_normal(0, 1, 1, 1) == 0.5


def test_case_2():
    assert kl_divergence_normal(0, 1, 0, 1) == 0
