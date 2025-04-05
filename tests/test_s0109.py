import numpy as np

from solutions.s0109_implement_layer_normalization_for_sequence_data import (
    layer_normalization,
)


def test_case_1():
    np.random.seed(42)

    X = np.random.randn(2, 2, 3)
    gamma = np.ones(3).reshape(1, 1, -1)
    beta = np.zeros(3).reshape(1, 1, -1)

    assert np.allclose(
        layer_normalization(X, gamma, beta),
        np.array(
            [
                [[0.474, -1.391, 0.917], [1.414, -0.707, -0.707]],
                [[1.132, 0.168, -1.3], [1.414, -0.705, -0.71]],
            ]
        ),
        atol=1e-4,
    )


def test_case_2():
    np.random.seed(42)

    X = np.random.randn(2, 3, 4)
    gamma = np.ones(4).reshape(1, 1, -1)
    beta = np.zeros(4).reshape(1, 1, -1)

    assert np.allclose(
        layer_normalization(X, gamma, beta),
        np.array(
            [
                [
                    [-0.229, -1.3, 0.026, 1.502],
                    [-0.926, -0.926, 1.46, 0.392],
                    [-0.585, 1.732, -0.571, -0.576],
                ],
                [
                    [1.401, -1.05, -0.836, 0.486],
                    [-0.4, 1.657, -0.238, -1.019],
                    [1.454, -0.191, 0.094, -1.357],
                ],
            ]
        ),
        atol=1e-4,
    )


def test_case_3():
    np.random.seed(42)

    X = np.random.randn(2, 3, 4)
    gamma = np.ones(4).reshape(1, 1, -1) * 0.5
    beta = np.ones(4).reshape(1, 1, -1)

    assert np.allclose(
        layer_normalization(X, gamma, beta),
        np.array(
            [
                [
                    [0.886, 0.35, 1.013, 1.751],
                    [0.537, 0.537, 1.73, 1.196],
                    [0.708, 1.866, 0.715, 0.712],
                ],
                [
                    [1.7, 0.475, 0.582, 1.243],
                    [0.8, 1.828, 0.881, 0.49],
                    [1.727, 0.904, 1.047, 0.322],
                ],
            ]
        ),
        atol=1e-4,
    )
