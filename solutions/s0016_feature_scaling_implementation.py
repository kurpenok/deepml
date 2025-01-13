import numpy as np


def feature_scaling(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    scaled_data = (data - mean) / std

    min_value = np.min(data, axis=0)
    max_value = np.max(data, axis=0)
    normalized_data = (data - min_value) / (max_value - min_value)

    return np.round(scaled_data, 4), np.round(normalized_data, 4)
