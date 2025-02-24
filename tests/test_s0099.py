from solutions.s0099_implement_the_softplus_activation_function import softplus


def test_case_1():
    assert softplus(2) == 2.1269


def test_case_2():
    assert softplus(0) == 0.6931


def test_case_3():
    assert softplus(100) == 100
