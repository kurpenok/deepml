import numpy as np


def adam_optimizer(
    parameter: float | np.ndarray,
    grad: float | np.ndarray,
    m: float | np.ndarray,
    v: float | np.ndarray,
    t: int,
    learning_rate: float = 0.001,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
) -> tuple[float | np.ndarray, float | np.ndarray, float | np.ndarray]:
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * (grad**2)

    m_hat = m / (1 - beta1**t)
    v_hat = v / (1 - beta2**t)

    update = learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
    parameter = parameter - update

    return np.round(parameter, 5), np.round(m, 5), np.round(v, 5)
