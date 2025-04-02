import numpy as np

from solutions.s0090_bm25_ranking import calculate_bm25_scores


def test_case_1():
    assert np.allclose(
        calculate_bm25_scores(
            [["the", "cat", "sat"], ["the", "dog", "ran"], ["the", "bird", "flew"]],
            ["the", "cat"],
        ),
        np.array([0.693, 0.0, 0.0]),
        atol=1e-4,
    )


def test_case_2():
    assert np.allclose(
        calculate_bm25_scores([["the"] * 10, ["the"]], ["the"]),
        np.array([0.0, 0.0]),
        atol=1e-4,
    )


def test_case_3():
    assert np.allclose(
        calculate_bm25_scores([["term"] * 10, ["the"] * 2], ["term"], k1=1.0),
        np.array([0.705, 0.0]),
        atol=1e-4,
    )
