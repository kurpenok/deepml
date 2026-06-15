def compressed_col_sparse_matrix(
    dense_matrix: list[list[int]],
) -> tuple[list[int], list[int], list[int]]:
    values = []
    row_indices = []
    column_pointer = [0]

    for j in range(len(dense_matrix[0])):
        for i in range(len(dense_matrix)):
            if dense_matrix[i][j]:
                values.append(dense_matrix[i][j])
                row_indices.append(i)
        column_pointer.append(len(values))

    return values, row_indices, column_pointer
