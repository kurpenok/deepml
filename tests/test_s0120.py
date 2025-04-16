from solutions.s0120_bhattacharyya_distance_between_two_distributions import (
    bhattacharyya_distance,
)


def test_case_1():
    assert bhattacharyya_distance([0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]) == 0.1166


def test_case_2():
    assert bhattacharyya_distance([0.7, 0.2, 0.1], [0.4, 0.3, 0.3]) == 0.0541


def test_case_3():
    assert bhattacharyya_distance([], [0.5, 0.4, 0.1]) == 0.0


def test_case_4():
    assert bhattacharyya_distance([0.6, 0.4], [0.1, 0.7, 0.2]) == 0.0


def test_case_5():
    assert bhattacharyya_distance([0.6, 0.2, 0.1, 0.1], [0.1, 0.2, 0.3, 0.4]) == 0.2007
