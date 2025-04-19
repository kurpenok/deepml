import numpy as np


def softplus(x: np.ndarray) -> np.ndarray:
    return np.log(1 + np.exp(x))


def softmax(x: np.ndarray) -> np.ndarray:
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)


def compute_topk_mask(H: np.ndarray, k: int) -> np.ndarray:
    mask = np.zeros_like(H, dtype=bool)
    for i in range(H.shape[0]):
        top_k_indices = np.argpartition(H[i], -k)[-k:]
        mask[i, top_k_indices] = True
    return mask


def noisy_topk_gating(
    X: np.ndarray, W_g: np.ndarray, W_noise: np.ndarray, N: np.ndarray, k: int
) -> np.ndarray:
    H_base = X @ W_g
    H_noise = X @ W_noise

    softplus_H = softplus(H_noise)

    H = H_base + N * softplus_H

    mask = compute_topk_mask(H, k)
    H_prime = np.where(mask, H, -np.inf)

    return softmax(H_prime)
