from solutions.s0121_vector_element_wise_sum import vector_sum


def test_case_1():
    assert vector_sum([1, 3], [4, 5]) == [5, 8]


def test_case_2():
    assert vector_sum([1, 2, 3], [4, 5, 6]) == [5, 7, 9]


def test_case_3():
    assert vector_sum([1, 2], [1, 2, 3]) == -1


def test_case_4():
    assert vector_sum([1.5, 2.5, 3.0], [2, 1, 4]) == [3.5, 3.5, 7.0]
