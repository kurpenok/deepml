import numpy as np

from src.s0024_single_neuron import single_neuron_model


def test_case_1():
    features = [[0.5, 1.0], [-1.5, -2.0], [2.0, 1.5]]
    labels = [0.0, 1.0, 0.0]
    weights = [0.7, -0.4]
    bias = -0.1

    result = single_neuron_model(features, labels, weights, bias)
    expected_probabilities = [0.4626, 0.4134, 0.6682]
    expected_mse = 0.3349

    assert np.allclose(result[0], expected_probabilities, atol=1e-4)
    assert result[1] == expected_mse


def test_case_2():
    features = [[1.0, 2.0], [2.0, 3.0], [3.0, 1.0]]
    labels = [1.0, 0.0, 1.0]
    weights = [0.5, -0.2]
    bias = 0.0

    result = single_neuron_model(features, labels, weights, bias)
    expected_probabilities = [0.525, 0.5987, 0.7858]
    expected_mse = 0.21

    assert np.allclose(result[0], expected_probabilities, atol=1e-4)
    assert result[1] == expected_mse
