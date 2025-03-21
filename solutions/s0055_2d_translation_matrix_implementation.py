import numpy as np


def translate_object(points: np.ndarray, tx: int, ty: int) -> np.ndarray:
    translation_matrix = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]])
    homogeneous_points = np.hstack([np.array(points), np.ones((len(points), 1))])

    return (homogeneous_points @ translation_matrix.T)[:, :2]
