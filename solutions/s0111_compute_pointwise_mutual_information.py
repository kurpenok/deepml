import math


def compute_pmi(
    joint_counts: int, total_counts_x: int, total_counts_y: int, total_samples: int
) -> float:
    p_x = total_counts_x / total_samples
    p_y = total_counts_y / total_samples
    p_xy = joint_counts / total_samples

    if p_xy == 0 or p_x == 0 or p_y == 0:
        return float("-inf")

    pmi = math.log2(p_xy / (p_x * p_y))
    if int(pmi) == pmi:
        return int(pmi)
    return round(pmi, 3)
