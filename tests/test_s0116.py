from solutions.s0116_derivative_of_a_polynomial import poly_term_derivative


def test_case_1():
    assert poly_term_derivative(2.0, 3.0, 2.0) == 12.0


def test_case_2():
    assert poly_term_derivative(1.5, 4.0, 0.0) == 0.0


def test_case_3():
    assert poly_term_derivative(3.0, 2.0, 3.0) == 36.0


def test_case_4():
    assert poly_term_derivative(0.5, 5.0, 1.0) == 0.5
