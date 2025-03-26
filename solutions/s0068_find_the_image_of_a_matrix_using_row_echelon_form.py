import numpy as np


def rref(A: np.ndarray) -> np.ndarray:
    A = A.astype(np.float32)

    for i in range(A.shape[0]):
        if A[i, i] == 0:
            nonzero_current_row = np.nonzero(A[i:, i])[0] + i
            if len(nonzero_current_row) == 0:
                continue
            A[[i, nonzero_current_row[0]]] = A[[nonzero_current_row[0], i]]

        A[i] = A[i] / A[i, i]

        for j in range(A.shape[0]):
            if i != j:
                A[j] -= A[i] * A[j, i]

    return A


def find_pivot_columns(A: np.ndarray) -> list[np.ndarray]:
    pivot_columns = []

    for i in range(A.shape[0]):
        nonzero = np.nonzero(A[i, :])[0]
        if len(nonzero):
            pivot_columns.append(nonzero[0])

    return pivot_columns


def matrix_image(A: np.ndarray) -> np.ndarray:
    Arref = rref(A)

    pivot_columns = find_pivot_columns(Arref)
    image_basis = A[:, pivot_columns]

    return image_basis
