import numpy as np

from solutions.s0058_gaussian_elimination_for_solving_linear_systems import (
    gaussian_elimination,
)


def test_case_1():
    assert np.allclose(
        gaussian_elimination(
            np.array([[2.0, 8.0, 4.0], [2.0, 5.0, 1.0], [4.0, 10.0, -1.0]]),
            np.array([2.0, 5.0, 1.0]),
        ),
        np.array([11.0, -4.0, 3.0]),
        atol=1e-4,
    )


def test_case_2():
    assert np.allclose(
        gaussian_elimination(
            np.array(
                [
                    [0.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                    [2.0, 6.0, 2.0, 1.0, 0.0, 0.0, 0.0],
                    [1.0, 2.0, 7.0, 2.0, 1.0, 0.0, 0.0],
                    [0.0, 1.0, 2.0, 8.0, 2.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0, 2.0, 9.0, 2.0, 1.0],
                    [0.0, 0.0, 0.0, 1.0, 2.0, 10.0, 2.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 11.0],
                ]
            ),
            np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]),
        ),
        np.array(
            [
                -0.4894027,
                0.36169985,
                0.2766003,
                0.25540569,
                0.31898951,
                0.40387497,
                0.53393278,
            ]
        ),
        atol=1e-4,
    )


def test_case_3():
    assert np.allclose(
        gaussian_elimination(
            np.array([[2.0, 1.0, -1.0], [-3.0, -1.0, 2.0], [-2.0, 1.0, 2.0]]),
            np.array([8.0, -11.0, -3.0]),
        ),
        np.array([2.0, 3.0, -1.0]),
        atol=1e-4,
    )
