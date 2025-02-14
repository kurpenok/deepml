import numpy as np


def calculate_contrast(img: np.ndarray) -> int:
    return img.max() - img.min()
