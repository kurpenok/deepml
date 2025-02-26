import math


def swish(x: float) -> float:
    return round(x * (1 / (1 + math.exp(-x))), 4)
