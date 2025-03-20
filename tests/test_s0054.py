from solutions.s0054_implementing_a_simple_rnn import rnn_forward


def test_case_1():
    assert rnn_forward(
        [[1.0], [2.0], [3.0]],
        [0.0],
        [[0.5]],
        [[0.8]],
        [0.0],
    ) == [0.9758816208890569]


def test_case_2():
    assert rnn_forward([[0.5], [0.1], [-0.2]], [0.0], [[1.0]], [[0.5]], [0.1]) == [
        0.11795168176753276
    ]
