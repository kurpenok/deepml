from solutions.s0111_compute_pointwise_mutual_information import compute_pmi


def test_case_1():
    assert compute_pmi(50, 200, 300, 1000) == -0.263


def test_case_2():
    assert compute_pmi(10, 50, 50, 200) == -0.322


def test_case_3():
    assert compute_pmi(100, 500, 500, 1000) == -1.322


def test_case_4():
    assert compute_pmi(100, 400, 600, 1200) == -1
