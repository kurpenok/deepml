import numpy as np


class SimpleRNN:
    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        self.hidden_size = hidden_size
        self.W_xh = np.random.randn(hidden_size, input_size) * 0.01
        self.W_hh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.W_hy = np.random.randn(output_size, hidden_size) * 0.01
        self.b_h = np.zeros((hidden_size, 1))
        self.b_y = np.zeros((output_size, 1))

    def forward(self, x: np.ndarray) -> np.ndarray:
        T = x.shape[0]
        self.inputs = []
        self.hidden_states = [np.zeros((self.hidden_size, 1))]
        self.outputs = []

        h_prev = self.hidden_states[0]
        for t in range(T):
            x_t = x[t].reshape(-1, 1)
            self.inputs.append(x_t)

            h_current = np.tanh(self.W_xh @ x_t + self.W_hh @ h_prev + self.b_h)
            self.hidden_states.append(h_current)

            y_t = self.W_hy @ h_current + self.b_y
            self.outputs.append(y_t)

            h_prev = h_current

        outputs_array = np.concatenate([y.T for y in self.outputs], axis=0)
        return outputs_array

    def backward(self, x: np.ndarray, y: np.ndarray, learning_rate: float) -> None:
        T = x.shape[0]

        dW_xh = np.zeros_like(self.W_xh)
        dW_hh = np.zeros_like(self.W_hh)
        dW_hy = np.zeros_like(self.W_hy)
        db_h = np.zeros_like(self.b_h)
        db_y = np.zeros_like(self.b_y)

        d_prev_h = np.zeros((self.hidden_size, 1))

        for t in reversed(range(T)):
            y_pred_t = self.outputs[t]
            y_true_t = y[t].reshape(-1, 1)

            dy = y_pred_t - y_true_t

            h_t = self.hidden_states[t + 1]
            dW_hy += dy @ h_t.T
            db_y += dy

            dh_t = self.W_hy.T @ dy + d_prev_h

            dh_raw = dh_t * (1 - h_t**2)

            x_t = self.inputs[t]
            dW_xh += dh_raw @ x_t.T

            h_prev = self.hidden_states[t]
            dW_hh += dh_raw @ h_prev.T

            db_h += dh_raw

            d_prev_h = self.W_hh.T @ dh_raw

        self.W_xh -= learning_rate * dW_xh
        self.W_hh -= learning_rate * dW_hh
        self.W_hy -= learning_rate * dW_hy
        self.b_h -= learning_rate * db_h
        self.b_y -= learning_rate * db_y
