import numpy as np

from solutions.s0025_single_neuron_with_backpropagation import train_neuron


def test_case_1():
    result = train_neuron(
        np.array([[1.0, 2.0], [2.0, 1.0], [-1.0, -2.0]]),
        np.array([1, 0, 0]),
        np.array([0.1, -0.2]),
        0.0,
        0.1,
        2,
    )

    assert np.allclose(result[0], np.array([0.1036, -0.1425]), atol=1e-4)
    assert result[1] == -0.0167
    assert np.allclose(result[2], np.array([0.3033, 0.2942]), atol=1e-4)


def test_case_2():
    result = train_neuron(
        np.array([[1, 2], [2, 3], [3, 1]]),
        np.array([1, 0, 1]),
        np.array([0.5, -0.2]),
        0,
        0.1,
        3,
    )

    assert np.allclose(result[0], np.array([0.4892, -0.2301]), atol=1e-4)
    assert result[1] == 0.0029
    assert np.allclose(result[2], np.array([0.21, 0.2087, 0.2076]), atol=1e-4)
