import numpy as np

from solutions.s0113_implement_a_simple_residual_block_with_shortcut_connection import (
    residual_block,
)


def test_case_1():
    assert np.allclose(
        residual_block(
            np.array([1.0, 2.0]),
            np.array([[1.0, 0.0], [0.0, 1.0]]),
            np.array([[0.5, 0.0], [0.0, 0.5]]),
        ),
        np.array([1.5, 3.0]),
        atol=1e-4,
    )


def test_case_2():
    assert np.allclose(
        residual_block(
            np.array([-1.0, 2.0]),
            np.array([[1.0, 0.0], [0.0, 1.0]]),
            np.array([[0.5, 0.0], [0.0, 0.5]]),
        ),
        np.array([0.0, 3.0]),
        atol=1e-4,
    )


def test_case_3():
    assert np.allclose(
        residual_block(
            np.array([0.0, 0.0]),
            np.array([[1.0, 0.0], [0.0, 1.0]]),
            np.array([[0.5, 0.0], [0.0, 0.5]]),
        ),
        np.array([0.0, 0.0]),
        atol=1e-4,
    )
