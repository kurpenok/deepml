import numpy as np


def gini_impurity(y: np.ndarray) -> float:
    return 1 - np.sum(np.unique(y, return_counts=True)[1] ** 2) / y.size**2
