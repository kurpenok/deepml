import numpy as np


def gauss_seidel_iter(A: np.ndarray, b: np.ndarray, x: np.ndarray) -> np.ndarray:
    rows, cols = A.shape

    for i in range(rows):
        x_new = b[i]

        for j in range(cols):
            if i != j:
                x_new -= A[i, j] * x[j]

        x[i] = x_new / A[i, i]

    return x


def gauss_seidel(
    A: np.ndarray, b: np.ndarray, n: int, x_ini: np.ndarray | None = None
) -> np.ndarray:
    x = x_ini or np.zeros_like(b)

    for _ in range(n):
        x = gauss_seidel_iter(A, b, x)

    return x
