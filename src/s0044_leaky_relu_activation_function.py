def leaky_relu(z: float, alpha: float = 0.01) -> float:
    if z > 0:
        return z
    return alpha * z
