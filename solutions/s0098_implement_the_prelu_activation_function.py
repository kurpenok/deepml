def prelu(x: float, alpha: float = 0.25) -> float:
    return float(x) if x > 0 else float(round(alpha * x, 3))
