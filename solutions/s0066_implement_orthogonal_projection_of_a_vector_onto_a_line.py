import numpy as np


def orthogonal_projection(v: np.ndarray, l: np.ndarray) -> np.ndarray:
    np_v = np.array(v)
    np_l = np.array(l)
    return np.array(
        [round(point, 3) for point in ((np_v @ np_l) / (np_l @ np_l)) * np_l]
    )
