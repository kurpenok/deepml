def calculate_covariance_matrix(vectors: list[list[float]]) -> list[list[float]]:
    n_features = len(vectors)
    if not n_features:
        return []

    n_observations = len(vectors[0])
    if not n_observations:
        return [[0.0] * n_observations for _ in range(n_observations)]

    means = [sum(feature) / n_observations for feature in vectors]

    covariance_matrix = [[0.0] * n_features for _ in range(n_features)]

    for observation_i in range(n_observations):
        dev = [vectors[i][observation_i] - means[i] for i in range(n_features)]
        for i in range(n_features):
            for j in range(n_features):
                covariance_matrix[i][j] += dev[i] * dev[j]

    if n_observations > 1:
        for i in range(n_features):
            for j in range(n_features):
                covariance_matrix[i][j] /= n_observations - 1

    return covariance_matrix
