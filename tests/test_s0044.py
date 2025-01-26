from solutions.s0044_leaky_relu_activation_function import leaky_relu


def test_case_1():
    assert leaky_relu(0) == 0


def test_case_2():
    assert leaky_relu(1) == 1


def test_case_3():
    assert leaky_relu(-1) == -0.01


def test_case_4():
    assert leaky_relu(-2, 0.1) == -0.2
