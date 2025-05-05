import numpy as np


def train_neuron(
    features: np.ndarray,
    labels: np.ndarray,
    initial_weights: np.ndarray,
    initial_bias: float,
    learning_rate: float,
    epochs: int,
) -> tuple[np.ndarray, float, list[float]]:
    n_samples = features.shape[0]
    weights = initial_weights.copy()
    bias = initial_bias
    mse_values = []

    for _ in range(epochs):
        z = np.dot(features, weights) + bias
        a = 1 / (1 + np.exp(-z))

        mse = np.mean((a - labels) ** 2)
        mse_rounded = round(mse, 4)
        mse_values.append(mse_rounded)

        delta = (a - labels) * a * (1 - a) * (2 / n_samples)
        grad_weights = np.dot(features.T, delta)
        grad_bias = np.sum(delta)

        weights -= learning_rate * grad_weights
        bias -= learning_rate * grad_bias

    updated_weights = np.round(weights, 4)
    updated_bias = round(bias, 4)

    return updated_weights, updated_bias, mse_values
