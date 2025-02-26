from solutions.s0102_implement_the_swish_activation_function import swish


def test_case_1():
    assert swish(1) == 0.7311


def test_case_2():
    assert swish(0) == 0
