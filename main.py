from solutions.s0001_matrix_times_vector import matrix_dot_vector
from solutions.s0002_transpose_of_a_matrix import transpose_matrix
from solutions.s0003_reshape_matrix import reshape_matrix
from solutions.s0004_calculate_mean_by_row_or_column import calculate_matrix_mean


# 0001. Matrix times Vector
assert matrix_dot_vector([[1, 2], [2, 4]], [1, 2]) == [5, 10]
assert matrix_dot_vector([[1, 2], [2, 4], [6, 8], [12, 4]], [1, 2, 3]) == -1

# 0002. Transpose of a Matrix
assert transpose_matrix([[1, 2], [3, 4], [5, 6]]) == [[1, 3, 5], [2, 4, 6]]
assert transpose_matrix([[1, 2, 3], [4, 5, 6]]) == [[1, 4], [2, 5], [3, 6]]

# 0003. Reshape Matrix
assert reshape_matrix([[1, 2, 3, 4], [5, 6, 7, 8]], (4, 2)) == [
    [1, 2],
    [3, 4],
    [5, 6],
    [7, 8],
]
assert reshape_matrix([[1, 2, 3], [4, 5, 6]], (3, 2)) == [[1, 2], [3, 4], [5, 6]]

# 0004. Calculate Mean by Row or Column
assert calculate_matrix_mean([[1, 2, 3], [4, 5, 6], [7, 8, 9]], "column") == [
    4.0,
    5.0,
    6.0,
]
assert calculate_matrix_mean([[1, 2, 3], [4, 5, 6], [7, 8, 9]], "row") == [
    2.0,
    5.0,
    8.0,
]
