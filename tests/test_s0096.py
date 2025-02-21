from solutions.s0096_implement_hard_sigmoid_activation_function import hard_sigmoid


def test_case_1():
    assert hard_sigmoid(0.0) == 0.5


def test_case_2():
    assert hard_sigmoid(3.0) == 1.0
