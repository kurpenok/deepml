from solutions.s0023_softmax_activation_function_implementation import softmax


def test_case_1():
    assert softmax([1, 2, 3]) == [0.0900, 0.2447, 0.6652]


def test_case_2():
    assert softmax([1, 1, 1]) == [0.3333, 0.3333, 0.3333]
