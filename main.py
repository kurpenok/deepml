from tasks.t0001_matrix_times_vector import matrix_dot_vector


# 0001. Matrix times Vector
assert matrix_dot_vector([[1, 2], [2, 4]], [1, 2]) == [5, 10]
assert matrix_dot_vector([[1, 2], [2, 4], [6, 8], [12, 4]], [1, 2, 3]) == -1
