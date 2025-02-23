from solutions.s0098_implement_the_prelu_activation_function import prelu


def test_case_1():
    assert prelu(-2.0, 0.25) == -0.5


def test_case_2():
    assert prelu(2) == 2


def test_case_3():
    assert prelu(0) == 0
