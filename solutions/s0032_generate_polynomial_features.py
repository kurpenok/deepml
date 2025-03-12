from itertools import combinations_with_replacement

import numpy as np


def polynomial_features(X: np.ndarray, degree: int) -> np.ndarray:
    n_samples, n_features = np.shape(X)

    combinations = [
        item
        for sublist in [
            combinations_with_replacement(range(n_features), i)
            for i in range(0, degree + 1)
        ]
        for item in sublist
    ]

    n_output_features = len(combinations)
    X_new = np.empty((n_samples, n_output_features))

    for i, index_combs in enumerate(combinations):
        X_new[:, i] = np.prod(X[:, index_combs], axis=1)

    return X_new
