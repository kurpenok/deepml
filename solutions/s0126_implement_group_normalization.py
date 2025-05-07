import numpy as np


def group_normalization(
    X: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
    num_groups: int,
    epsilon: float = 1e-5,
) -> np.ndarray:
    B, C, H, W = X.shape
    if C % num_groups != 0:
        raise ValueError("Number of channels must be divisible by num_groups")

    group_size = C // num_groups
    X_reshaped = X.reshape(B, num_groups, group_size, H, W)

    mu = X_reshaped.mean(axis=(2, 3, 4), keepdims=True)
    var = X_reshaped.var(axis=(2, 3, 4), keepdims=True)

    X_normalized_reshaped = (X_reshaped - mu) / np.sqrt(var + epsilon)

    X_normalized = X_normalized_reshaped.reshape(B, C, H, W)

    Y = gamma[None, :, None, None] * X_normalized + beta[None, :, None, None]

    return Y
