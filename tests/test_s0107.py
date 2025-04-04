import numpy as np

from solutions.s0107_implement_masked_self_attention import (
    compute_qkv,
    masked_attention,
)


def test_case_1():
    np.random.seed(42)

    X = np.arange(48).reshape(6, 8)
    X = np.random.permutation(X.flatten()).reshape(6, 8)

    mask = np.triu(np.ones((6, 6)) * (-np.inf), k=1)
    W_q = np.random.randint(0, 4, size=(8, 8))
    W_k = np.random.randint(0, 5, size=(8, 8))
    W_v = np.random.randint(0, 6, size=(8, 8))

    assert (
        masked_attention(*compute_qkv(X, W_q, W_k, W_v), mask)
        == np.array(
            [
                [547, 490, 399, 495, 485, 439, 645, 393],
                [547, 490, 399, 495, 485, 439, 645, 393],
                [471, 472, 429, 538, 377, 450, 531, 362],
                [471, 472, 429, 538, 377, 450, 531, 362],
                [471, 472, 429, 538, 377, 450, 531, 362],
                [471, 472, 429, 538, 377, 450, 531, 362],
            ]
        )
    ).all()


def test_case_2():
    np.random.seed(42)

    X = np.arange(16).reshape(4, 4)
    X = np.random.permutation(X.flatten()).reshape(4, 4)

    mask = np.triu(np.ones((4, 4)) * (-np.inf), k=1)
    W_q = np.random.randint(0, 4, size=(4, 4))
    W_k = np.random.randint(0, 5, size=(4, 4))
    W_v = np.random.randint(0, 6, size=(4, 4))

    assert (
        masked_attention(*compute_qkv(X, W_q, W_k, W_v), mask)
        == np.array(
            [
                [52, 63, 48, 71],
                [103, 109, 46, 99],
                [103, 109, 46, 99],
                [103, 109, 46, 99],
            ]
        )
    ).all()
