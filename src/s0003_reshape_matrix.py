def reshape_matrix(
    matrix: list[list[int | float]], new_shape: tuple[int, int]
) -> list[list[int | float]]:
    new_rows = new_shape[0]
    new_cols = new_shape[1]

    if matrix and len(matrix) * len(matrix[0]) != new_rows * new_cols:
        return []

    flatten_a = [n for row in matrix for n in row]

    return [flatten_a[i * new_cols : (i + 1) * new_cols] for i in range(new_rows)]
