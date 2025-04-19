import numpy as np

from solutions.s0124_implement_the_noisy_top_k_gating_function import noisy_topk_gating


def test_case_1():
    assert np.allclose(
        noisy_topk_gating(
            np.array([[1.0, 2.0]]),
            np.array([[1.0, 0.0], [0.0, 1.0]]),
            np.array([[0.5, 0.5], [0.5, 0.5]]),
            np.array([[1.0, -1.0]]),
            2,
        ),
        np.array([[0.917, 0.083]]),
        atol=1e-4,
    )


def test_case_2():
    assert np.allclose(
        noisy_topk_gating(
            np.array([[1.0, 2.0]]),
            np.array([[1.0, 0.0], [0.0, 1.0]]),
            np.zeros((2, 2)),
            np.zeros((1, 2)),
            1,
        ),
        np.array([[0.0, 1.0]]),
        atol=1e-4,
    )
