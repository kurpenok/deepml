import numpy as np

from solutions.s0087_adam_optimizer import adam_optimizer


def test_case_1():
    assert adam_optimizer(1.0, 0.1, 0.0, 0.0, 1) == (0.999, 0.01, 0.00001)


def test_case_2():
    result = adam_optimizer(
        np.array([1.0, 2.0]), np.array([0.1, 0.2]), np.zeros(2), np.zeros(2), 1
    )

    assert np.allclose(result[0], np.array([0.999, 1.999]), atol=0.01)
    assert np.allclose(result[1], np.array([0.01, 0.02]), atol=0.01)
    assert np.allclose(result[2], np.array([1.0e-05, 4.0e-05]), atol=0.01)
