import numpy as np


def orthonormal_basis(
    vectors: list[list[float]], tol: float = 1e-10
) -> list[np.ndarray]:
    basis = []

    for v in vectors:
        current = np.array(v, dtype=np.float64)
        for u in basis:
            proj = (current @ u) * u
            current -= proj

        norm = np.linalg.norm(current)
        if norm > tol:
            basis.append(current / norm)

    return basis
