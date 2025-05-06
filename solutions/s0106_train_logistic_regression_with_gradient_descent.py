import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))


def train_logreg(
    X: np.ndarray, y: np.ndarray, learning_rate: float, iterations: int
) -> tuple[np.ndarray, np.ndarray]:
    n_samples, n_features = X.shape
    X_aug = np.hstack([np.ones((n_samples, 1)), X])

    beta = np.zeros(n_features + 1)

    loss_history = []

    for _ in range(iterations):
        z = X_aug @ beta
        p = sigmoid(z)

        epsilon = 1e-15
        p_clipped = np.clip(p, epsilon, 1 - epsilon)
        loss = -np.sum(y * np.log(p_clipped) + (1 - y) * np.log(1 - p_clipped))
        loss_rounded = np.round(loss, 4)
        loss_history.append(loss_rounded)

        gradient = X_aug.T @ (p - y)

        beta -= learning_rate * gradient

    return np.round(beta, 4), np.array(loss_history)
