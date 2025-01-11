import numpy as np


def linear_regression_normal_equation(
    X: list[list[float]], y: list[float]
) -> list[float]:
    npx = np.array(X)
    npy = np.array(y).reshape(-1, 1)

    theta = np.linalg.inv(npx.T @ npx) @ npx.T @ npy
    theta = np.round(theta, 4).flatten().tolist()

    return theta
