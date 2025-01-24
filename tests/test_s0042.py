from solutions.s0042_implement_relu_activation_function import relu


def test_case_1():
    assert relu(0) == 0


def test_case_2():
    assert relu(1) == 1


def test_case_3():
    assert relu(-1) == 0
