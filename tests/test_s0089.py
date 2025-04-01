import numpy as np

from solutions.s0089_the_pattern_weavers_code import pattern_weaver


def test_case_1():
    assert np.allclose(
        pattern_weaver(5, np.array([4, 2, 7, 1, 9]), 1),
        np.array([8.9993, 8.9638, 9.0, 8.7259, 9.0]),
        atol=1e-4,
    )


def test_case_2():
    assert np.allclose(
        pattern_weaver(3, np.array([1, 3, 5]), 1),
        np.array([4.7019, 4.995, 4.9999]),
        atol=1e-4,
    )


def test_case_3():
    assert np.allclose(
        pattern_weaver(4, np.array([2, 8, 6, 4]), 1),
        np.array([7.9627, 8.0, 8.0, 7.9993]),
        atol=1e-4,
    )
