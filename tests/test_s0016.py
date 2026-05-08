import numpy as np

from src.s0016_feature_scaling_implementation import feature_scaling


def test_case_1():
    data = np.array([[1, 2], [3, 4], [5, 6]])

    result = feature_scaling(data)
    expected_standardized_data = np.array(
        [[-1.2247, -1.2247], [0.0, 0.0], [1.2247, 1.2247]]
    )
    expected_normalized_data = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])

    assert np.allclose(result[0], expected_standardized_data, atol=1e-4)
    assert np.allclose(result[1], expected_normalized_data, atol=1e-4)


def test_case_2():
    data = np.array([[0, 0], [1, 10], [2, 20]])

    result = feature_scaling(data)
    expected_standardized_data = np.array(
        [[-1.2247, -1.2247], [0.0, 0.0], [1.2247, 1.2247]]
    )
    expected_normalized_data = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])

    assert np.allclose(result[0], expected_standardized_data, atol=1e-4)
    assert np.allclose(result[1], expected_normalized_data, atol=1e-4)


def test_case_3():
    data = np.array([[-1, -2], [0, 0], [1, 2]])

    result = feature_scaling(data)
    expected_standardized_data = np.array(
        [[-1.2247, -1.2247], [0.0, 0.0], [1.2247, 1.2247]]
    )
    expected_normalized_data = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])

    assert np.allclose(result[0], expected_standardized_data, atol=1e-4)
    assert np.allclose(result[1], expected_normalized_data, atol=1e-4)


def test_case_4():
    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])

    result = feature_scaling(data)
    expected_standardized_data = np.array(
        [
            [-1.3416, -1.3416, -1.3416],
            [-0.4472, -0.4472, -0.4472],
            [0.4472, 0.4472, 0.4472],
            [1.3416, 1.3416, 1.3416],
        ]
    )
    expected_normalized_data = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.3333, 0.3333, 0.3333],
            [0.6667, 0.6667, 0.6667],
            [1.0, 1.0, 1.0],
        ]
    )

    assert np.allclose(result[0], expected_standardized_data, atol=1e-4)
    assert np.allclose(result[1], expected_normalized_data, atol=1e-4)
