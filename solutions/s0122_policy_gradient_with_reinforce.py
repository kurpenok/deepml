import numpy as np


def compute_policy_gradient(
    theta: np.ndarray, episodes: list[list[tuple[int, int, float]]]
) -> np.ndarray:
    _, num_actions = theta.shape
    gradient = np.zeros_like(theta)

    for episode in episodes:
        T = len(episode)
        G = 0

        for t in range(T - 1, -1, -1):
            s_t, a_t, r_t = episode[t]
            G += r_t

            log_policy_gradient = np.zeros_like(theta)
            for a in range(num_actions):
                if a == a_t:
                    log_policy_gradient[s_t, a] = 1 - np.exp(theta[s_t, a]) / np.sum(
                        np.exp(theta[s_t, :])
                    )
                else:
                    log_policy_gradient[s_t, a] = -np.exp(theta[s_t, a]) / np.sum(
                        np.exp(theta[s_t, :])
                    )

            gradient += log_policy_gradient * G

    gradient /= len(episodes)

    return gradient
