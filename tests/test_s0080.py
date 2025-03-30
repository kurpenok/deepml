from solutions.s0080_normal_distribution_pdf_calculator import normal_pdf


def test_case_1():
    assert normal_pdf(16, 15, 2.04) == 0.17342


def test_case_2():
    assert normal_pdf(0, 0, 1) == 0.39894


def test_case_3():
    assert normal_pdf(1, 0, 0.5) == 0.10798
