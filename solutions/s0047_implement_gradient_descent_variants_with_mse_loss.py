import numpy as np


def gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    learning_rate: float,
    n_iterations: int,
    batch_size: int = 1,
    method: str = "batch",
):
    m = y.shape[0]

    for _ in range(n_iterations):
        if method == "batch":
            predictions = X.dot(weights)
            errors = predictions - y
            gradient = 2 * X.T.dot(errors) / m
            weights = weights - learning_rate * gradient

        elif method == "stochastic":
            for i in range(m):
                prediction = X[i].dot(weights)
                error = prediction - y[i]
                gradient = 2 * X[i].T.dot(error)
                weights = weights - learning_rate * gradient

        elif method == "mini_batch":
            for i in range(0, m, batch_size):
                X_batch = X[i : i + batch_size]
                y_batch = y[i : i + batch_size]
                predictions = X_batch.dot(weights)
                errors = predictions - y_batch
                gradient = 2 * X_batch.T.dot(errors) / batch_size
                weights = weights - learning_rate * gradient

    return np.round(weights, 1)
