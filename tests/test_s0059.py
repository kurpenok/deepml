import numpy as np

from solutions.s0059_implement_long_short_term_memory_network import LSTM


def test_case_1():
    input_sequence = np.array([[1.0], [2.0], [3.0]])
    initial_hidden_state = np.zeros((1, 1))
    initial_cell_state = np.zeros((1, 1))

    lstm = LSTM(input_size=1, hidden_size=1)

    lstm.Wf = np.array([[0.5, 0.5]])
    lstm.Wi = np.array([[0.5, 0.5]])
    lstm.Wc = np.array([[0.3, 0.3]])
    lstm.Wo = np.array([[0.5, 0.5]])

    lstm.bf = np.array([[0.1]])
    lstm.bi = np.array([[0.1]])
    lstm.bc = np.array([[0.1]])
    lstm.bo = np.array([[0.1]])

    assert np.allclose(
        lstm.forward(input_sequence, initial_hidden_state, initial_cell_state)[1],
        [[0.73698596]],
        atol=1e-4,
    )


def test_case_2():
    input_sequence = np.array([[0.1, 0.2], [0.3, 0.4]])
    initial_hidden_state = np.zeros((2, 1))
    initial_cell_state = np.zeros((2, 1))

    lstm = LSTM(input_size=2, hidden_size=2)

    lstm.Wf = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]])
    lstm.Wi = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]])
    lstm.Wc = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]])
    lstm.Wo = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]])

    lstm.bf = np.array([[0.1], [0.2]])
    lstm.bi = np.array([[0.1], [0.2]])
    lstm.bc = np.array([[0.1], [0.2]])
    lstm.bo = np.array([[0.1], [0.2]])

    assert np.allclose(
        lstm.forward(input_sequence, initial_hidden_state, initial_cell_state)[1],
        [[0.16613133], [0.40299449]],
        atol=1e-4,
    )
