from src.s0022_sigmoid_activation_function_understanding import sigmoid


def test_case_1():
    assert sigmoid(0) == 0.5


def test_case_2():
    assert sigmoid(1) == 0.7311


def test_case_3():
    assert sigmoid(-1) == 0.2689
