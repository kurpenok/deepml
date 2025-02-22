from solutions.s0097_implement_the_elu_activation_function import elu


def test_case_1():
    assert elu(-1) == -0.6321


def test_case_2():
    assert elu(1) == 1
