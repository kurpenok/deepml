import numpy as np


def matrixmul(
    a: list[list[int | float]], b: list[list[int | float]]
) -> list[list[int | float]] | int:
    try:
        return [list(row) for row in list(np.array(a) @ np.array(b))]
    except:
        return -1
