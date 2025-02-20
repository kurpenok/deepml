from solutions.s0095_calculate_the_phi_coefficient import phi_corr


def test_case_1():
    assert phi_corr([1, 1, 0, 0], [0, 0, 1, 1]) == -1


def test_case_2():
    assert phi_corr([1, 1, 0, 0], [1, 0, 1, 1]) == -0.5774
