import numpy as np

from solutions.s0038_implement_ada_boost_fit_method import adaboost_fit


def test_case_1():
    assert adaboost_fit(
        np.array([[1, 2], [2, 3], [3, 4], [4, 5]]), np.array([1, 1, -1, -1]), 3
    ) == [
        {
            "polarity": -1,
            "threshold": 3,
            "feature_index": 0,
            "alpha": 11.512925464920228,
        },
        {
            "polarity": -1,
            "threshold": 3,
            "feature_index": 0,
            "alpha": 11.512925464920228,
        },
        {
            "polarity": -1,
            "threshold": 3,
            "feature_index": 0,
            "alpha": 11.512925464920228,
        },
    ]


def test_case_2():
    assert adaboost_fit(
        np.array(
            [
                [8, 7],
                [3, 4],
                [5, 9],
                [4, 0],
                [1, 0],
                [0, 7],
                [3, 8],
                [4, 2],
                [6, 8],
                [0, 2],
            ]
        ),
        np.array([1, -1, 1, -1, 1, -1, -1, -1, 1, 1]),
        2,
    ) == [
        {
            "polarity": 1,
            "threshold": 5,
            "feature_index": 0,
            "alpha": 0.6931471805599453,
        },
        {
            "polarity": -1,
            "threshold": 3,
            "feature_index": 0,
            "alpha": 0.5493061443340549,
        },
    ]
