from solutions.s0100_implement_the_softsign_activation_function import softsign


def test_case_1():
    assert softsign(1) == 0.5


def test_case_2():
    assert softsign(0) == 0
