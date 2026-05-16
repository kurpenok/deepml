import numpy as np

from src.s0039_implementation_of_log_softmax_function import log_softmax


def test_case_1():
    scores = np.array([1, 2, 3])

    result = log_softmax(scores)
    expected = np.array([-2.4076, -1.4076, -0.4076])

    assert np.allclose(result, expected)


def test_case_2():
    scores = np.array([1, 1, 1])

    result = log_softmax(scores)
    expected = np.array([-1.0986, -1.0986, -1.0986])

    assert np.allclose(result, expected)
