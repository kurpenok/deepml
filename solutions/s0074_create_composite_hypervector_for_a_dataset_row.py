import numpy as np


def create_hv(dim: int) -> np.ndarray:
    return np.random.choice([-1, 1], dim)


def create_col_hvs(dim: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    np.random.seed(seed)
    return create_hv(dim), create_hv(dim)


def bind(hv1: np.ndarray, hv2: np.ndarray) -> np.ndarray:
    return hv1 * hv2


def bundle(hvs: dict[str, np.ndarray]) -> np.ndarray:
    bundled = np.sum(list(hvs.values()), axis=0)
    return sign(bundled)


def sign(vector: np.ndarray) -> np.ndarray:
    return np.array([1 if v >= 0 else -1 for v in vector])


def create_row_hv(
    row: dict[str, str], dim: int, random_seeds: dict[str, int]
) -> np.ndarray:
    row_hvs = {col: bind(*create_col_hvs(dim, random_seeds[col])) for col in row.keys()}
    return bundle(row_hvs)
