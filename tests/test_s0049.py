import numpy as np

from solutions.s0049_implement_adam_optimization_algorithm import (
    adam_optimizer,
    gradient,
    objective_function,
)


def test_case_1():
    assert np.allclose(
        adam_optimizer(objective_function, gradient, np.array([1.0, 1.0])),
        np.array([0.99000325, 0.99000325]),
        atol=1e-4,
    )


def test_case_2():
    assert np.allclose(
        adam_optimizer(objective_function, gradient, np.array([0.2, 12.3])),
        np.array([0.19001678, 12.29000026]),
        atol=1e-4,
    )
