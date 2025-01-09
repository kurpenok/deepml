def transpose_matrix(a: list[list[int | float]]) -> list[list[int | float]]:
    num_rows = len(a)
    num_cols = len(a[0])

    transposed_matrix = [[0.0 for _ in range(num_rows)] for _ in range(num_cols)]

    for i in range(num_rows):
        for j in range(num_cols):
            transposed_matrix[j][i] = a[i][j]

    return transposed_matrix
