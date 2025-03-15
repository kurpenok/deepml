import numpy as np


def simple_conv2d(
    input_matrix: np.ndarray, kernel: np.ndarray, padding: int, stride: int
) -> np.ndarray:
    input_matrix = np.pad(input_matrix, padding, "constant")

    input_height, input_width = input_matrix.shape
    kernel_height, kernel_width = kernel.shape

    output_matrix = []

    for i in range(0, input_height, stride):
        output_row = []

        for j in range(0, input_width, stride):
            if i + kernel_height <= input_height and j + kernel_width <= input_width:
                submatrix = input_matrix[i : i + kernel_height, j : j + kernel_width]
                output_row.append(np.sum(submatrix * kernel))

        if output_row:
            output_matrix.append(output_row[:])

    return np.array(np.round(output_matrix, 1))
