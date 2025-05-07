from solutions.s0127_find_captain_redbeards_hidden_treasure import find_treasure


def test_case_1():
    assert find_treasure(0.0) == 0.0


# def test_case_2():
#     assert abs(find_treasure(-1) - 2.25) < 1e-2


def test_case_3():
    assert abs(find_treasure(1.0) - 2.25) < 1e-2


def test_case_4():
    assert abs(find_treasure(3.0) - 2.25) < 1e-2
