import math


def poisson_probability(k: int, lam: float):
    return round((lam**k * math.exp(-lam)) / (math.factorial(k)), 5)
