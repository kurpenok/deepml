import numpy as np


def transform_basis(B: list[list[float]], C: list[list[float]]) -> list[list[float]]:
    return (np.linalg.inv(np.array(C)) @ np.array(B)).tolist()
