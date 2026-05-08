import math


def softmax(scores: list[float]) -> list[float]:
    return [
        round(
            math.exp(z - max(scores)) / sum(math.exp(z - max(scores)) for z in scores),
            4,
        )
        for z in scores
    ]
