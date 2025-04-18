from solutions.s0123_calculate_computational_efficiency_of_moe import compute_efficiency


def test_case_1():
    assert compute_efficiency(1000, 2, 512, 512) == 99.8


def test_case_2():
    assert compute_efficiency(10, 2, 256, 256) == 80.0
