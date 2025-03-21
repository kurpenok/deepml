import numpy as np

from solutions.s0055_2d_translation_matrix_implementation import translate_object


def test_case_1():
    assert (
        translate_object(np.array([[0, 0], [1, 0], [0.5, 1]]), 2, 3)
        == np.array([[2.0, 3.0], [3.0, 3.0], [2.5, 4.0]])
    ).all()


def test_case_2():
    assert (
        translate_object(np.array([[0, 0], [1, 0], [1, 1], [0, 1]]), -1, 2)
        == np.array([[-1.0, 2.0], [0.0, 2.0], [0.0, 3.0], [-1.0, 3.0]])
    ).all()
