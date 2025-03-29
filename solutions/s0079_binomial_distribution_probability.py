import math


def binomial_probability(n: int, k: int, p: float) -> float:
    return round(math.comb(n, k) * (p**k) * ((1 - p) ** (n - k)), 5)
