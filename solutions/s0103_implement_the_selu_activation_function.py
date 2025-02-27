import math


def selu(x: float) -> float:
    alpha = 1.6732632423543772
    scale = 1.0507009873554804
    return round(scale * x, 4) if x > 0 else round(scale * alpha * (math.exp(x) - 1), 4)
