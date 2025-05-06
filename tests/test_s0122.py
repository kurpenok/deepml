import numpy as np

from solutions.s0122_policy_gradient_with_reinforce import compute_policy_gradient


def test_case_1():
    assert np.allclose(
        compute_policy_gradient(
            np.zeros((2, 2)), episodes=[[(0, 1, 0), (1, 0, 1)], [(0, 0, 0)]]
        ),
        np.array([[-0.25, 0.25], [0.25, -0.25]]),
        atol=1e-4,
    )


def test_case_2():
    assert np.allclose(
        compute_policy_gradient(
            np.zeros((2, 2)), [[(0, 0, 0)], [(0, 1, 0), (1, 1, 0)]]
        ),
        np.array([[0.0, 0.0], [0.0, 0.0]]),
        atol=1e-4,
    )
