from collections import Counter

import numpy as np


def descriptive_statistics(data: list[float]) -> dict[str, float]:
    np_data = np.array(data)

    mean = np.mean(np_data)
    median = np.median(np_data)
    mode = Counter(np_data).most_common(1)[0][0]
    variance = np.sum((np_data - np.mean(np_data)) ** 2) / np_data.shape[0]
    std_dev = np.sqrt(variance)
    percentiles = [np.percentile(np_data, p) for p in (25, 50, 75)]
    iqr = percentiles[2] - percentiles[0]

    stats_dict = {
        "mean": mean,
        "median": median,
        "mode": mode,
        "variance": np.round(variance, 4),
        "standard_deviation": np.round(std_dev, 4),
        "25th_percentile": percentiles[0],
        "50th_percentile": percentiles[1],
        "75th_percentile": percentiles[2],
        "interquartile_range": iqr,
    }

    return stats_dict
