import numpy as np


def conjugate_gradient(
    A: np.ndarray,
    b: np.ndarray,
    n: int,
    tol: float = 1e-8,
) -> np.ndarray:
    x = np.zeros_like(b)
    r = residual(A, b, x)
    r_plus1 = r
    p = r

    for _ in range(n):
        alp = alpha(A, r, p)

        x = x + alp * p
        r_plus1 = r - alp * (A @ p)

        bet = beta(r, r_plus1)

        p = r_plus1 + bet * p
        r = r_plus1

        if np.linalg.norm(residual(A, b, x)) < tol:
            break

    return x


def residual(A: np.ndarray, b: np.ndarray, x: np.ndarray) -> np.ndarray:
    return b - A @ x


def alpha(A: np.ndarray, r: np.ndarray, p: np.ndarray) -> float:
    alpha_num = np.dot(r, r)
    alpha_den = np.dot(p @ A, p)
    return alpha_num / alpha_den


def beta(r: np.ndarray, r_plus1: np.ndarray) -> float:
    beta_num = np.dot(r_plus1, r_plus1)
    beta_den = np.dot(r, r)
    return beta_num / beta_den
