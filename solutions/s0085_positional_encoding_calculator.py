import numpy as np


def pos_encoding(position: int, d_model: int) -> float | np.ndarray:
    if position == 0 or d_model <= 0:
        return -1

    pos = np.arange(position).reshape(-1, 1)
    i = np.arange(d_model)

    div_term = 10000.0 ** (2 * (i // 2) / d_model)
    angle = pos / div_term

    pe = np.zeros((position, d_model))
    even_mask = (i % 2) == 0

    pe[:, even_mask] = np.sin(angle[:, even_mask])
    pe[:, ~even_mask] = np.cos(angle[:, ~even_mask])

    pe = pe[np.newaxis, ...].astype(np.float16)
    return pe
