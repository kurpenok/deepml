import math


def normal_pdf(x: int, mean: int, std_dev: float) -> float:
    return round(
        (1 / (math.sqrt(2 * math.pi * std_dev**2)))
        * (math.exp(-((x - mean) ** 2) / (2 * std_dev**2))),
        5,
    )
