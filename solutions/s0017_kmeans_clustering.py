import numpy as np


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    return np.sqrt(((a - b) ** 2).sum(axis=1))


def k_means_clustering(
    points: list[tuple],
    k: int,
    initial_centroids: list[tuple],
    max_iterations: int,
) -> list[tuple[float, float]]:
    np_points = np.array(points)
    centroids = np.array(initial_centroids)

    for _ in range(max_iterations):
        distances = np.array(
            [euclidean_distance(np_points, centroid) for centroid in centroids]
        )
        assignments = np.argmin(distances, axis=0)

        new_centroids = np.array(
            [
                np_points[assignments == i].mean(axis=0)
                if len(np_points[assignments == i]) > 0
                else centroids[i]
                for i in range(k)
            ]
        )

        if np.all(centroids == new_centroids):
            break

        centroids = new_centroids
        centroids = np.round(centroids, 4)

    return [tuple(centroid) for centroid in centroids]
