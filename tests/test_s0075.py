from solutions.s0075_generate_a_confusion_matrix_for_binary_classification import (
    confusion_matrix,
)


def test_case_1():
    assert confusion_matrix([[1, 1], [1, 0], [0, 1], [0, 0], [0, 1]]) == [
        [1, 1],
        [2, 1],
    ]


def test_case_2():
    assert confusion_matrix(
        [
            [0, 1],
            [1, 0],
            [1, 1],
            [0, 1],
            [0, 0],
            [1, 0],
            [0, 1],
            [1, 1],
            [0, 0],
            [1, 0],
            [1, 1],
            [0, 0],
            [1, 0],
            [0, 1],
            [1, 1],
            [1, 1],
            [1, 0],
        ]
    ) == [[5, 5], [4, 3]]
