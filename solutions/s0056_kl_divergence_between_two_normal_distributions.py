import math


def kl_divergence_normal(
    mu_p: float, sigma_p: float, mu_q: float, sigma_q: float
) -> float:
    return (
        (math.log(sigma_q / sigma_p))
        + ((sigma_p**2 + (mu_p - mu_q) ** 2) / (2 * sigma_q**2))
        - 0.5
    )
