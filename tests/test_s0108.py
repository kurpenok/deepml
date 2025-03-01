from solutions.s0108_measure_disorder_in_apple_colors import disorder


def test_case_1():
    assert disorder([1, 1, 0, 0]) == 0.5


def test_case_2():
    assert disorder([0, 0, 0, 0]) < disorder([1, 0, 0, 0])


def test_case_3():
    assert disorder([0, 0, 0, 0, 0, 1, 2, 3]) < disorder([0, 0, 1, 1, 2, 2, 3, 3])
