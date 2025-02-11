import numpy as np

from solutions.s0076_calculate_cosine_similarity_between_vectors import (
    cosine_similarity,
)


def test_case_1():
    assert cosine_similarity(np.array([1, 2, 3]), np.array([2, 4, 6])) == 1


def test_case_2():
    assert cosine_similarity(np.array([1, 2, 3]), np.array([-1, -2, -3])) == -1
