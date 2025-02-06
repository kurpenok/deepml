from solutions.s0070_calculate_image_brightness import calculate_brightness


def test_case_1():
    assert calculate_brightness([[100, 200], [50, 150]]) == 125


def test_case_2():
    assert calculate_brightness([]) == -1


def test_case_3():
    assert calculate_brightness([[100, 200], [150]]) == -1
