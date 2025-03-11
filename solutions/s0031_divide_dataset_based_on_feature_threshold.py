import numpy as np


def divide_on_feature(
    X: np.ndarray, feature_i: int, threshold: int
) -> list[list[np.ndarray]]:
    new_features = [[], []]

    for i in range(X.shape[0]):
        if X[i][feature_i] >= threshold:
            new_features[0].append(X[i].copy())
        else:
            new_features[1].append(X[i].copy())

    return new_features
