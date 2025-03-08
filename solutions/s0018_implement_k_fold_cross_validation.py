import numpy as np


def k_fold_cross_validation(
    X: np.ndarray,
    y: np.ndarray,
    k: int = 5,
    shuffle: bool = True,
    random_seed: int | None = None,
) -> list[tuple[list, list]]:
    n_samples = X.shape[0]
    indices = np.arange(n_samples)

    if shuffle:
        np.random.seed(random_seed) if random_seed is not None else None
        np.random.shuffle(indices)

    fold_sizes = np.full(k, n_samples // k, dtype=int)
    fold_sizes[: n_samples % k] += 1

    folds = []
    current = 0
    for fold_size in fold_sizes:
        folds.append(indices[current : current + fold_size])
        current += fold_size

    return [
        (np.concatenate(folds[:i] + folds[i + 1 :]).tolist(), folds[i].tolist())
        for i in range(k)
    ]
