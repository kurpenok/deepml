import math
from typing import Any, Optional, Tuple

import numpy as np

# DO NOT CHANGE SEED
np.random.seed(42)


# DO NOT CHANGE LAYER CLASS
class Layer(object):
    def set_input_shape(self, shape):
        self.input_shape = shape

    def layer_name(self):
        return self.__class__.__name__

    def parameters(self):
        return 0

    def forward_pass(self, X, training):
        raise NotImplementedError()

    def backward_pass(self, accum_grad):
        raise NotImplementedError()

    def output_shape(self):
        raise NotImplementedError()


class Dense(Layer):
    def __init__(self, n_units: int, input_shape: Optional[Tuple[int, ...]] = None):
        self.layer_input: Optional[np.ndarray] = None
        self.input_shape = input_shape
        self.n_units = n_units
        self.trainable = True
        self.W: Optional[np.ndarray] = None
        self.w0: Optional[np.ndarray] = None
        self.optimizer: Optional[Any] = None

    def initialize(self, optimizer: Any) -> None:
        self.optimizer = optimizer
        limit = 1 / math.sqrt(self.input_shape[0])
        self.W = np.random.uniform(-limit, limit, (self.input_shape[0], self.n_units))
        self.w0 = np.zeros((1, self.n_units))

    def forward_pass(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        self.layer_input = X
        return X.dot(self.W) + self.w0

    def backward_pass(self, accum_grad: np.ndarray) -> np.ndarray:
        grad_W = self.layer_input.T.dot(accum_grad)
        grad_w0 = np.sum(accum_grad, axis=0, keepdims=True)

        grad_input = accum_grad.dot(self.W.T)

        if self.trainable:
            self.W = self.optimizer.update(self.W, grad_W)
            self.w0 = self.optimizer.update(self.w0, grad_w0)

        return grad_input

    def output_shape(self) -> Tuple[int]:
        return (self.n_units,)

    def parameters(self) -> int:
        return np.prod(self.W.shape) + np.prod(self.w0.shape)
