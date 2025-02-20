import numpy as np


def phi_corr(x: list[int], y: list[int]) -> float:
    np_x = np.array(x)
    np_y = np.array(y)

    x_00 = np.sum((np_x == 0) & (np_y == 0))
    x_01 = np.sum((np_x == 0) & (np_y == 1))
    x_10 = np.sum((np_x == 1) & (np_y == 0))
    x_11 = np.sum((np_x == 1) & (np_y == 1))

    return np.round(
        ((x_00 * x_11) - (x_01 * x_10))
        / np.sqrt((x_00 + x_01) * (x_10 + x_11) * (x_00 + x_10) * (x_01 + x_11)),
        4,
    )
