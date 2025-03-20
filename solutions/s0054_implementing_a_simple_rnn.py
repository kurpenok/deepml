import numpy as np


def rnn_forward(
    input_sequence: list[list[float]],
    initial_hidden_state: list[float],
    Wx: list[list[float]],
    Wh: list[list[float]],
    b: list[float],
) -> list[float]:
    np_input_sequence = np.array(input_sequence)
    np_initial_hidden_state = np.array(initial_hidden_state)
    np_Wx = np.array(Wx)
    np_Wh = np.array(Wh)
    np_b = np.array(b)

    h = np_initial_hidden_state

    for x in np_input_sequence:
        h = np.tanh((np_Wx @ x) + (np_Wh @ h) + np_b)

    return list(h)
