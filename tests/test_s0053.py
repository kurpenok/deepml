import numpy as np

from solutions.s0053_implement_self_attention_mechanism import (
    compute_qkv,
    self_attention,
)


def test_case_1():
    X = np.array([[1, 0], [0, 1]])
    W_q = np.array([[1, 0], [0, 1]])
    W_k = np.array([[1, 0], [0, 1]])
    W_v = np.array([[1, 2], [3, 4]])

    Q, K, V = compute_qkv(X, W_q, W_k, W_v)

    assert np.allclose(
        self_attention(Q, K, V),
        np.array([[1.660477, 2.660477], [2.339523, 3.339523]]),
        atol=1e-4,
    )


def test_case_2():
    X = np.array([[1, 1], [1, 0]])
    W_q = np.array([[1, 0], [0, 1]])
    W_k = np.array([[1, 0], [0, 1]])
    W_v = np.array([[1, 2], [3, 4]])

    Q, K, V = compute_qkv(X, W_q, W_k, W_v)

    assert np.allclose(
        self_attention(Q, K, V),
        np.array([[3.00928465, 4.6790462], [2.5, 4.0]]),
        atol=1e-4,
    )


def test_case_3():
    X = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0]])
    W_q = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]])
    W_k = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]])
    W_v = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    Q, K, V = compute_qkv(X, W_q, W_k, W_v)

    assert np.allclose(
        self_attention(Q, K, V),
        np.array(
            [
                [8.0, 10.0, 12.0],
                [8.61987385, 10.61987385, 12.61987385],
                [7.38012615, 9.38012615, 11.38012615],
            ]
        ),
        atol=1e-4,
    )
