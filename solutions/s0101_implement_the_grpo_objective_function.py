import numpy as np


def grpo_objective(
    rhos: list[float],
    A: list[float],
    pi_theta_old: list[float],
    pi_theta_ref: list[float],
    epsilon: float = 0.2,
    beta: float = 0.01,
) -> float:
    np_rhos = np.array(rhos)
    np_A = np.array(A)
    np_pi_theta_old = np.array(pi_theta_old)
    np_pi_theta_ref = np.array(pi_theta_ref)

    clipped_rhos = np.clip(np_rhos, 1 - epsilon, 1 + epsilon)

    min_terms = np.minimum(np_rhos * np_A, clipped_rhos * np_A)
    average_min_terms = np.mean(min_terms)

    new_policy = np_rhos * np_pi_theta_old
    sum_new = np.sum(new_policy)
    new_policy_normalized = new_policy / sum_new

    sum_ref = np.sum(np_pi_theta_ref)
    ref_policy_normalized = np_pi_theta_ref / sum_ref

    kl_divergence = np.sum(
        new_policy_normalized
        * (np.log(new_policy_normalized) - np.log(ref_policy_normalized))
    )

    objective = average_min_terms - beta * kl_divergence
    return round(objective, 6)
