import numpy as np


def l1_regularization_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float = 0.1,
    learning_rate: float = 0.01,
    max_iter: int = 1000,
    tol: float = 1e-4,
) -> tuple[np.ndarray, float]:
    n_samples, n_features = X.shape

    weights = np.zeros(n_features)
    bias = 0

    for _ in range(max_iter):
        y_pred = np.dot(X, weights) + bias
        loss = y_pred - y

        grad_w = (1 / n_samples) * (X.T @ loss) + alpha * np.sign(weights)
        grad_b = (1 / n_samples) * np.sum(loss)

        weights -= learning_rate * grad_w
        bias -= learning_rate * grad_b

        if np.linalg.norm(grad_w, ord=1) < tol:
            break

    return weights, bias
