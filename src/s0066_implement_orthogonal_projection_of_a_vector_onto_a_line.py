import numpy as np


def orthogonal_projection(v: list[float], L: list[float]) -> list[float]:
    np_v = np.array(v)
    np_L = np.array(L)
    return (((np_v @ np_L) / (np_L @ np_L)) * np_L).tolist()
