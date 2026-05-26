from src.s0056_kl_divergence_between_two_normal_distributions import (
    kl_divergence_normal,
)


def test_case_1():
    mu_p = 0.0
    sigma_p = 1.0
    mu_q = 0.0
    sigma_q = 1.0

    result = kl_divergence_normal(mu_p, sigma_p, mu_q, sigma_q)
    expected = 0.0

    assert result == expected


def test_case_2():
    mu_p = 0.0
    sigma_p = 1.0
    mu_q = 1.0
    sigma_q = 1.0

    result = kl_divergence_normal(mu_p, sigma_p, mu_q, sigma_q)
    expected = 0.5

    assert result == expected


def test_case_3():
    mu_p = 0.0
    sigma_p = 1.0
    mu_q = 0.0
    sigma_q = 2.0

    result = kl_divergence_normal(mu_p, sigma_p, mu_q, sigma_q)
    expected = 0.3181471805599453

    assert result == expected


def test_case_4():
    mu_p = 1.0
    sigma_p = 1.0
    mu_q = 0.0
    sigma_q = 2.0

    result = kl_divergence_normal(mu_p, sigma_p, mu_q, sigma_q)
    expected = 0.4431471805599453

    assert result == expected
