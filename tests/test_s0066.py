import numpy as np

from solutions.s0066_implement_orthogonal_projection_of_a_vector_onto_a_line import (
    orthogonal_projection,
)


def test_case_1():
    assert (orthogonal_projection(np.array([3, 4]), np.array([1, 0])) == [3, 0]).all()


def test_case_2():
    assert (
        orthogonal_projection(np.array([1, 2, 3]), np.array([0, 0, 1])) == [0, 0, 3]
    ).all()
