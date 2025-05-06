import numpy as np


def moe(
    x: np.ndarray, We: np.ndarray, Wg: np.ndarray, n_experts: int, top_k: int
) -> np.ndarray:
    n_batch, l_seq, d_model = x.shape
    x_flat = x.reshape(-1, d_model)

    logits = x_flat @ Wg

    max_logits = np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits - max_logits)
    sum_exp = np.sum(exp_logits, axis=1, keepdims=True)
    alpha = exp_logits / sum_exp

    sorted_indices = np.argsort(-alpha, axis=1)
    top_k_indices = sorted_indices[:, :top_k]

    alpha_values = np.take_along_axis(alpha, top_k_indices, axis=1)
    sum_alpha = np.sum(alpha_values, axis=1, keepdims=True)
    renormalized_alpha = alpha_values / sum_alpha

    tilde_alpha = np.zeros_like(alpha)
    np.put_along_axis(tilde_alpha, top_k_indices, renormalized_alpha, axis=1)

    output = np.zeros_like(x_flat)
    for i in range(n_experts):
        expert_contribution = x_flat @ We[i]
        output += expert_contribution * tilde_alpha[:, i, np.newaxis]

    return output.reshape(n_batch, l_seq, d_model)
