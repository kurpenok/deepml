import math


def softmax(scores: list[float]) -> list[float]:
    return [round(math.exp(z) / sum([math.exp(z) for z in scores]), 4) for z in scores]
