import numpy as np


def get_random_subsets(
    X: np.ndarray,
    y: np.ndarray,
    n_subsets: int,
    replacements: bool = True,
    seed: int = 42,
) -> list[tuple[list[list[float]], list[float]]]:
    np.random.seed(seed)

    n, _ = X.shape

    subset_size = n if replacements else n // 2
    idx = np.array(
        [
            np.random.choice(n, subset_size, replace=replacements)
            for _ in range(n_subsets)
        ]
    )

    return [(X[idx][i].tolist(), y[idx][i].tolist()) for i in range(n_subsets)]
