import numpy as np


class LSTM:
    def __init__(self, input_size: int, hidden_size: int):
        self.input_size: int = input_size
        self.hidden_size: int = hidden_size

        self.Wf: np.ndarray = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wi: np.ndarray = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wc: np.ndarray = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wo: np.ndarray = np.random.randn(hidden_size, input_size + hidden_size)

        self.bf: np.ndarray = np.zeros((hidden_size, 1))
        self.bi: np.ndarray = np.zeros((hidden_size, 1))
        self.bc: np.ndarray = np.zeros((hidden_size, 1))
        self.bo: np.ndarray = np.zeros((hidden_size, 1))

    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    def forward(
        self,
        x: np.ndarray,
        initial_hidden_state: np.ndarray,
        initial_cell_state: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        h = initial_hidden_state
        c = initial_cell_state
        outputs = []

        for t in range(len(x)):
            xt = x[t].reshape(-1, 1)
            concat = np.vstack((h, xt))

            ft = self.sigmoid((self.Wf @ concat) + self.bf)

            it = self.sigmoid((self.Wi @ concat) + self.bi)
            c_tilde = np.tanh((self.Wc @ concat) + self.bc)

            c = ft * c + it * c_tilde

            ot = self.sigmoid((self.Wo @ concat) + self.bo)

            h = ot * np.tanh(c)

            outputs.append(h)

        return np.array(outputs), h, c
