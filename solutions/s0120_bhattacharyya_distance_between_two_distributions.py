import numpy as np


def bhattacharyya_distance(p: list[float], q: list[float]) -> float:
    if not len(p) or not len(q) or len(p) != len(q):
        return 0.0
    return round(-np.log(sum(np.sqrt(pi * qi) for pi, qi in zip(p, q))), 4)
