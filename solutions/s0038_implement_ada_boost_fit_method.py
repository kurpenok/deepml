import numpy as np


def adaboost_fit(X: np.ndarray, y: np.ndarray, n_clf: int) -> list[dict[str, int]]:
    n_samples, n_features = X.shape
    w = np.full(n_samples, 1.0 / n_samples)
    classifiers = []

    for _ in range(n_clf):
        best_error = np.inf
        best_feature = 0
        best_threshold = 0.0
        best_polarity = 1

        for feature_idx in range(n_features):
            feature_values = X[:, feature_idx]
            unique_values = np.unique(feature_values)
            sorted_values = np.sort(unique_values)

            if len(sorted_values) > 1:
                thresholds = sorted_values[:-1]
            else:
                thresholds = sorted_values

            for threshold in thresholds:
                pred = np.where(feature_values >= threshold, 1, -1)
                error = np.sum(w * (pred != y))

                if error > 0.5:
                    error = 1.0 - error
                    polarity = -1
                else:
                    polarity = 1

                if error < best_error:
                    best_error = error
                    best_feature = feature_idx
                    best_threshold = threshold
                    best_polarity = polarity

        epsilon = 1e-10
        best_error = max(best_error, epsilon)
        best_error = min(best_error, 1 - epsilon)

        alpha = 0.5 * np.log((1.0 - best_error) / best_error)

        pred = best_polarity * np.where(X[:, best_feature] >= best_threshold, 1, -1)
        w *= np.exp(-alpha * y * pred)
        w /= np.sum(w)

        classifiers.append(
            {
                "polarity": best_polarity,
                "threshold": best_threshold,
                "feature_index": best_feature,
                "alpha": alpha,
            }
        )

    return classifiers
