from typing import Callable

import numpy as np


def objective_function(x: np.ndarray) -> float:
    return x[0] ** 2 + x[1] ** 2


def gradient(x: np.ndarray) -> np.ndarray:
    return np.array([2 * x[0], 2 * x[1]])


def adam_optimizer(
    f: Callable,
    grad: Callable,
    x0: np.ndarray,
    learning_rate: float = 0.001,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
    num_iterations: int = 10,
) -> np.ndarray:
    x = x0.copy()
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    t = 0

    for _ in range(num_iterations):
        t += 1
        g = grad(x)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * (g**2)
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        x -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

    return x
