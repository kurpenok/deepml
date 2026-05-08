import numpy as np

from src.s0023_softmax_activation_function_implementation import softmax


def test_case_1():
    scores = [1.0, 2.0, 3.0]

    result = softmax(scores)
    expected = [0.0900, 0.2447, 0.6652]

    assert np.allclose(result, expected, atol=1e-4)


def test_case_2():
    scores = [1.0, 1.0, 1.0]

    result = softmax(scores)
    expected = [0.3333, 0.3333, 0.3333]

    assert np.allclose(result, expected, atol=1e-4)


def test_case_3():
    scores = [-1.0, 0.0, 5.0]

    result = softmax(scores)
    expected = [0.0025, 0.0067, 0.9909]

    assert np.allclose(result, expected, atol=1e-4)


def test_case_4():
    scores = [1000.0, 2000.0, 3000.0]

    result = softmax(scores)
    expected = [0.0, 0.0, 1.0]

    assert np.allclose(result, expected, atol=1e-4)
