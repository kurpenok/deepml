from solutions.s0017_kmeans_clustering import k_means_clustering


def test_case_1():
    assert k_means_clustering(
        [(1, 2), (1, 4), (1, 0), (10, 2), (10, 4), (10, 0)], 2, [(1, 1), (10, 1)], 10
    ) == [(1.0, 2.0), (10.0, 2.0)]


def test_case_2():
    assert k_means_clustering(
        [(0, 0, 0), (2, 2, 2), (1, 1, 1), (9, 10, 9), (10, 11, 10), (12, 11, 12)],
        2,
        [(1, 1, 1), (10, 10, 10)],
        10,
    ) == [(1.0, 1.0, 1.0), (10.3333, 10.6667, 10.3333)]
