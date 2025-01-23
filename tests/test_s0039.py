from solutions.s0039_implementation_of_log_softmax_function import log_softmax


def test_case_1():
    assert log_softmax([1, 2, 3]) == [-2.4076, -1.4076, -0.4076]


def test_case_2():
    assert log_softmax([1, 1, 1]) == [-1.0986, -1.0986, -1.0986]
