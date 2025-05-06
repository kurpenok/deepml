import numpy as np

from solutions.s0128_dynamic_tanh_normalization_free_transformer_activation import (
    dynamic_tanh,
)


def test_case_1():
    assert np.allclose(
        dynamic_tanh(
            np.array([[[0.14115588, 0.00372817, 0.24126647, 0.22183601]]]),
            0.5,
            np.ones((4,)),
            np.zeros((4,)),
        ),
        np.array([[[0.0705, 0.0019, 0.1201, 0.1105]]]),
        atol=1e-4,
    )


def test_case_2():
    assert np.allclose(
        dynamic_tanh(
            np.array(
                [
                    [[0.94378259]],
                    [[0.97754654]],
                    [[0.36168351]],
                    [[0.51821078]],
                    [[0.76961589]],
                ]
            ),
            0.5,
            np.ones((1,)),
            np.zeros((1,)),
        ),
        np.array([[[0.4397]], [[0.4532]], [[0.1789]], [[0.2535]], [[0.3669]]]),
        atol=1e-4,
    )


def test_case_3():
    assert np.allclose(
        dynamic_tanh(
            np.array(
                [
                    [
                        [0.20793482, 0.16989285, 0.03898972],
                        [0.17912554, 0.10962205, 0.3870742],
                        [0.00107181, 0.35807922, 0.15861333],
                    ]
                ]
            ),
            0.5,
            np.ones((3,)),
            np.zeros((3,)),
        ),
        np.array(
            [
                [
                    [
                        [0.1036, 0.0847, 0.0195],
                        [0.0893, 0.0548, 0.1912],
                        [0.0005, 0.1772, 0.0791],
                    ]
                ]
            ]
        ),
        atol=1e-4,
    )
