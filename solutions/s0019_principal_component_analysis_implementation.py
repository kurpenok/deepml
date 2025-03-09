import numpy as np


def pca(data: np.ndarray, k: int) -> np.ndarray:
    data = (data - data.mean(axis=0)) / data.std(axis=0)

    cov = np.cov(data, rowvar=False)
    v, w = np.linalg.eig(cov)

    return np.round(w[:, v.argsort()[::-1]][:, :k], 4)
