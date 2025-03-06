import numpy as np


def solve_jacobi(A: np.ndarray, b: np.ndarray, n: int) -> list[float]:
    m = A.shape[0]

    x = [0] * m
    x_hold = [0] * m

    for _ in range(n):
        for i in range(m):
            x_hold[i] = (1 / A[i][i]) * (
                b[i] - sum([A[i][j] * x[j] for j in range(m) if i != j])
            )
        x = x_hold.copy()

    return [round(root, 4) for root in x]
