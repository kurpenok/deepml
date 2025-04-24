import numpy as np


def linear_kernel(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return x @ y


def rbf_kernel(x: np.ndarray, y: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    return np.exp(-(np.linalg.norm(x - y) ** 2) / (2 * (sigma**2)))


def pegasos_kernel_svm(
    data: np.ndarray,
    labels: np.ndarray,
    kernel: str = "linear",
    lambda_val: float = 0.01,
    iterations: int = 100,
    sigma: float = 1.0,
) -> tuple[list[float], float]:
    n_samples = len(data)
    alphas = np.zeros(n_samples)
    b = 0

    for t in range(1, iterations + 1):
        for i in range(n_samples):
            eta = 1.0 / (lambda_val * t)

            kernel_func = None
            if kernel == "linear":
                kernel_func = linear_kernel
            elif kernel == "rbf":
                kernel_func = lambda x, y: rbf_kernel(x, y, sigma)

            if kernel_func:
                decision = (
                    sum(
                        alphas[j] * labels[j] * kernel_func(data[j], data[i])
                        for j in range(n_samples)
                    )
                    + b
                )

                if labels[i] * decision < 1:
                    alphas[i] += eta * (labels[i] - lambda_val * alphas[i])
                    b += eta * labels[i]

    return [round(a, 4) for a in alphas], round(b, 4)
