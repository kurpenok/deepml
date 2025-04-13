import numpy as np

from solutions.s0115_implement_batch_normalization_for_bchw_input import (
    batch_normalization,
)


def test_case_1():
    np.random.seed(42)

    B, C, H, W = 2, 2, 2, 2
    X = np.random.randn(B, C, H, W)
    gamma = np.ones(C).reshape(1, C, 1, 1)
    beta = np.zeros(C).reshape(1, C, 1, 1)

    assert np.allclose(
        batch_normalization(X, gamma, beta),
        np.array(
            [
                [
                    [[0.42859934, -0.51776438], [0.65360963, 1.95820707]],
                    [[0.02353721, 0.02355215], [1.67355207, 0.93490043]],
                ],
                [
                    [[-1.01139563, 0.49692747], [-1.00236882, -1.00581468]],
                    [[0.45676349, -1.50433085], [-1.33293647, -0.27503802]],
                ],
            ],
        ),
        atol=1e-4,
    )


def test_case_2():
    np.random.seed(101)

    B, C, H, W = 2, 2, 2, 2
    X = np.random.randn(B, C, H, W)
    gamma = np.ones(C).reshape(1, C, 1, 1)
    beta = np.zeros(C).reshape(1, C, 1, 1)

    assert np.allclose(
        batch_normalization(X, gamma, beta),
        np.array(
            [
                [
                    [[1.81773164, 0.16104096], [0.38406453, 0.06197112]],
                    [[1.00432932, -0.37139956], [-1.12098938, 0.94031919]],
                ],
                [
                    [[-1.94800122, 0.25029395], [0.08188579, -0.80898678]],
                    [[0.34878049, -0.99452891], [-1.24171594, 1.43520478]],
                ],
            ]
        ),
        atol=1e-4,
    )


def test_case_3():
    np.random.seed(101)

    B, C, H, W = 2, 2, 2, 2
    X = np.random.randn(B, C, H, W)
    gamma = np.ones(C).reshape(1, C, 1, 1) * 0.5
    beta = np.ones(C).reshape(1, C, 1, 1)

    assert np.allclose(
        batch_normalization(X, gamma, beta),
        np.array(
            [
                [
                    [[1.90886582, 1.08052048], [1.19203227, 1.03098556]],
                    [[1.50216466, 0.81430022], [0.43950531, 1.4701596]],
                ],
                [
                    [[0.02599939, 1.12514697], [1.04094289, 0.59550661]],
                    [[1.17439025, 0.50273554], [0.37914203, 1.71760239]],
                ],
            ]
        ),
        atol=1e-4,
    )
