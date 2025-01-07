from tasks.t0001_matrix_times_vector import matrix_dot_vector
from tasks.t0002_transpose_of_a_matrix import transpose_matrix
from tasks.t0003_reshape_matrix import reshape_matrix


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
