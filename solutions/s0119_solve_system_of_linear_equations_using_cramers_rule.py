import numpy as np


def cramers_rule(A: list[list[float]], b: list[float]) -> int | list[float]:
    np_A = np.array(A)
    np_b = np.array(b)

    det_A = np.linalg.det(np_A)
    if np.abs(det_A) < 1e-12:
        return -1

    n = np_A.shape[0]
    x = np.zeros(n)

    for i in range(n):
        Ai = np_A.copy()
        Ai[:, i] = np_b
        det_Ai = np.linalg.det(Ai)
        x[i] = det_Ai / det_A

    return [x_i for x_i in np.round(x, 4)]
