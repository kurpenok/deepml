import math


def log_softmax(scores: list[float]) -> list[float]:
    return [
        round(
            scores[i]
            - max(scores)
            - math.log(
                sum([math.exp(scores[j] - max(scores)) for j in range(len(scores))])
            ),
            4,
        )
        for i in range(len(scores))
    ]
