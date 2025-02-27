from solutions.s0103_implement_the_selu_activation_function import selu


def test_case_1():
    assert selu(-1) == -1.1113


def test_case_2():
    assert selu(1) == 1.0507


def test_case_3():
    assert selu(5) == 5.2535
