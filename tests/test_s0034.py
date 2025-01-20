import numpy as np

from solutions.s0034_one_hot_encoding_of_normal_values import to_categorical


def test_case_1():
    assert (
        to_categorical(np.array([0, 1, 2, 1, 0]))
        == [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0],
        ]
    ).all()


def test_case_2():
    assert (
        to_categorical(np.array([3, 1, 2, 1, 3]), 4)
        == [
            [0, 0, 0, 1],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ]
    ).all()
