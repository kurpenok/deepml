from solutions.s0003_reshape_matrix import reshape_matrix


def test_case_1():
    assert reshape_matrix([[1, 2, 3, 4], [5, 6, 7, 8]], (4, 2)) == [
        [1, 2],
        [3, 4],
        [5, 6],
        [7, 8],
    ]


def test_case_2():
    assert reshape_matrix([[1, 2, 3], [4, 5, 6]], (3, 2)) == [[1, 2], [3, 4], [5, 6]]
