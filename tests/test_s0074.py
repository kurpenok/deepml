import numpy as np

from solutions.s0074_create_composite_hypervector_for_a_dataset_row import create_row_hv


def test_case_1():
    assert (
        create_row_hv(
            {"FeatureA": "value1", "FeatureB": "value2"},
            5,
            {"FeatureA": 42, "FeatureB": 7},
        )
        == np.array([1, -1, 1, 1, 1])
    ).all()


def test_case_2():
    assert (
        create_row_hv(
            {"FeatureA": "value1", "FeatureB": "value2"},
            10,
            {"FeatureA": 42, "FeatureB": 7},
        )
        == np.array([1, -1, 1, 1, -1, -1, -1, -1, -1, -1])
    ).all()


def test_case_3():
    assert (
        create_row_hv(
            {"FeatureA": "value1", "FeatureB": "value2"},
            15,
            {"FeatureA": 42, "FeatureB": 7},
        )
        == np.array([1, 1, -1, -1, 1, 1, 1, 1, -1, 1, 1, 1, -1, -1, 1])
    ).all()
