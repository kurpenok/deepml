import numpy as np


def compute_qkv(
    X: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return X @ W_q, X @ W_k, X @ W_v


def self_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    d_k = K.shape[1]
    scores = (Q @ K.T) / np.sqrt(d_k)

    attention_weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    attention_weights /= attention_weights.sum(axis=-1, keepdims=True)

    return attention_weights @ V
