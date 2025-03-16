import numpy as np

from solutions.s0047_implement_gradient_descent_variants_with_mse_loss import (
    gradient_descent,
)


def test_case_1():
    X = np.array([[1, 1], [2, 1], [3, 1], [4, 1]])
    y = np.array([2, 3, 4, 5])
    weights = np.zeros(X.shape[1])

    learning_rate = 0.01
    n_iterations = 1000
    batch_size = 2

    assert (
        gradient_descent(X, y, weights, learning_rate, n_iterations, method="batch")
        == np.array([1, 1])
    ).all()

    assert (
        gradient_descent(
            X, y, weights, learning_rate, n_iterations, method="stochastic"
        )
        == np.array([1, 1])
    ).all()

    assert (
        gradient_descent(
            X, y, weights, learning_rate, n_iterations, batch_size, method="mini_batch"
        )
        == np.array([1, 1])
    ).all()
