import math


def elu(x: float, alpha: float = 1.0) -> float:
    return float(x) if x > 0 else float(round(alpha * (math.exp(x) - 1), 4))
