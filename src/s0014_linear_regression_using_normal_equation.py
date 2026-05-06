import numpy as np


def linear_regression_normal_equation(
    X: list[list[float]], y: list[float]
) -> list[float]:
    np_X = np.array(X)
    np_y = np.array(y).reshape(-1, 1)

    theta = np.linalg.inv(np_X.T @ np_X) @ np_X.T @ np_y
    theta = np.round(theta, 4).flatten().tolist()

    return theta
