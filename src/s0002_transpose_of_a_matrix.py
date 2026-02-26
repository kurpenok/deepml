def transpose_matrix(a: list[list[int | float]]) -> list[list[int | float]]:
    if not len(a):
        return []

    num_cols = len(a)
    num_rows = len(a[0])

    transposed_a = [[0.0 for _ in range(num_cols)] for _ in range(num_rows)]

    for j in range(num_cols):
        for i in range(num_rows):
            transposed_a[i][j] = a[j][i]

    return transposed_a
