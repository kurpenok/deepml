import numpy as np


def feature_scaling(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    std = np.where(std != 0, std, 1.0)
    standardized_data = (data - mean) / std

    min_value = np.min(data, axis=0)
    max_value = np.max(data, axis=0)
    diff = max_value - min_value
    diff = np.where(diff != 0, diff, 1.0)
    normalized_data = (data - min_value) / (max_value - min_value)
    normalized_data[..., diff == 0] = 0.0

    return standardized_data, normalized_data
