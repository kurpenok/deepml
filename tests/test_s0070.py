from src.s0070_calculate_image_brightness import calculate_brightness


def test_case_1():
    img = []

    result = calculate_brightness(img)
    expected = -1

    assert result == expected


def test_case_2():
    img = [[100, 200], [50]]

    result = calculate_brightness(img)
    expected = -1

    assert result == expected


def test_case_3():
    img = [[100, 300]]

    result = calculate_brightness(img)
    expected = -1

    assert result


def test_case_4():
    img = [[128]]

    result = calculate_brightness(img)
    expected = 128

    assert result == expected


def test_case_5():
    img = [[100, 200], [50, 150]]

    result = calculate_brightness(img)
    expected = 125

    assert result == expected
