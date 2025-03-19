from solutions.s0051_optimal_string_alignment_distance import OSA


def test_case_1():
    assert OSA("butterfly", "dragonfly") == 6


def test_case_2():
    assert OSA("caper", "acer") == 2
