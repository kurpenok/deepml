import numpy as np

from solutions.s0063_implement_the_conjugate_gradient_method_for_solving_linear_systems import (
    conjugate_gradient,
)


def test_case_1():
    assert np.allclose(
        conjugate_gradient(np.array([[4, 1], [1, 3]]), np.array([1, 2]), 5),
        np.array([0.09090909, 0.63636364]),
        atol=1e-4,
    )


def test_case_2():
    assert np.allclose(
        conjugate_gradient(
            np.array([[4, 1, 2], [1, 3, 0], [2, 0, 5]]), np.array([7, 8, 5]), 1
        ),
        np.array([1.2627451, 1.44313725, 0.90196078]),
        atol=1e-4,
    )


def test_case_3():
    assert np.allclose(
        conjugate_gradient(
            np.array(
                [
                    [6, 2, 1, 1, 0],
                    [2, 5, 2, 1, 1],
                    [1, 2, 6, 1, 2],
                    [1, 1, 1, 7, 1],
                    [0, 1, 2, 1, 8],
                ]
            ),
            np.array([1, 2, 3, 4, 5]),
            100,
        ),
        np.array([0.01666667, 0.11666667, 0.21666667, 0.45, 0.5]),
        atol=1e-4,
    )
