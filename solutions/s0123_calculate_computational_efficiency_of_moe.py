def compute_efficiency(n_experts: int, k_active: int, d_in: int, d_out: int) -> float:
    flops_dense = n_experts * d_in * d_out
    flops_moe = k_active * d_in * d_out
    return (flops_dense - flops_moe) / flops_dense * 100
