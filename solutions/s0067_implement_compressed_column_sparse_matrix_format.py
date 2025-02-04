def compressed_col_sparse_matrix(
    dense_matrix: list[list[float]],
) -> tuple[list[float], list[int], list[int]]:
    elements = []
    row_indices = []
    column_pointers = [0]

    for j in range(len(dense_matrix[0])):
        for i in range(len(dense_matrix)):
            if dense_matrix[i][j]:
                elements.append(dense_matrix[i][j])
                row_indices.append(i)
        column_pointers.append(len(elements))

    return elements, row_indices, column_pointers
