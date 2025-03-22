import numpy as np

from solutions.s0057_gauss_seidel_method_for_solving_linear_systems import gauss_seidel


def test_case_1():
    assert np.allclose(
        gauss_seidel(
            np.array([[4.0, 1.0, 2.0], [3.0, 5.0, 1.0], [1.0, 1.0, 3.0]]),
            np.array([4.0, 7.0, 3.0]),
            100,
        ),
        np.array([0.5, 1.0, 0.5]),
        atol=1e-4,
    )


def test_case_2():
    assert np.allclose(
        gauss_seidel(
            np.array(
                [
                    [4.0, -1.0, 0.0, 1.0],
                    [-1.0, 4.0, -1.0, 0.0],
                    [0.0, -1.0, 4.0, -1.0],
                    [1.0, 0.0, -1.0, 4.0],
                ]
            ),
            np.array([15.0, 10.0, 10.0, 15.0]),
            1,
        ),
        np.array([3.75, 3.4375, 3.359375, 3.65234375]),
        atol=1e-4,
    )


def test_case_3():
    assert np.allclose(
        gauss_seidel(
            np.array([[10.0, -1.0, 2.0], [-1.0, 11.0, -1.0], [2.0, -1.0, 10.0]]),
            np.array([6.0, 25.0, -11.0]),
            100,
        ),
        np.array([1.04326923, 2.26923077, -1.08173077]),
        atol=1e-4,
    )
