import numpy as np

from solutions.s0016_feature_scaling_implementation import feature_scaling


def test_case_1():
    scaled_data, normalized_data = feature_scaling(np.array([[1, 2], [3, 4], [5, 6]]))

    expected_scaled_data = np.array([[-1.2247, -1.2247], [0.0, 0.0], [1.2247, 1.2247]])
    expected_normalized_data = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])

    assert np.all(np.isclose(scaled_data, expected_scaled_data, atol=1e-4))
    assert np.all(np.isclose(normalized_data, expected_normalized_data, atol=1e-4))


# def test_case_2():
