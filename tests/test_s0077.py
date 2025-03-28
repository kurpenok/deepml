from solutions.s0077_calculate_performance_metrics_for_a_classification_model import (
    performance_metrics,
)


def test_case_1():
    assert performance_metrics([1, 0, 1, 0, 1], [1, 0, 0, 1, 1]) == (
        [[2, 1], [1, 1]],
        0.6,
        0.667,
        0.5,
        0.5,
    )


def test_case_2():
    assert performance_metrics(
        [1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1],
        [0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0],
    ) == ([[6, 4], [2, 7]], 0.684, 0.667, 0.778, 0.636)


def test_case_3():
    assert performance_metrics(
        [0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0],
        [1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1],
    ) == ([[4, 4], [5, 2]], 0.4, 0.471, 0.286, 0.333)
