from solutions.s0024_single_neuron import single_neuron_model


def test_case_1():
    assert single_neuron_model(
        [[0.5, 1.0], [-1.5, -2.0], [2.0, 1.5]], [0, 1, 0], [0.7, -0.4], -0.1
    ) == ([0.4626, 0.4134, 0.6682], 0.3349)


def test_case_2():
    assert single_neuron_model([[1, 2], [2, 3], [3, 1]], [1, 0, 1], [0.5, -0.2], 0) == (
        [0.525, 0.5987, 0.7858],
        0.21,
    )
