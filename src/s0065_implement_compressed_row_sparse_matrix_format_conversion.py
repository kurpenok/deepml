def compressed_row_sparse_matrix(
    dense_matrix: list[list[int]],
) -> tuple[list[int], list[int], list[int]]:
    non_zero_values = []
    column_indices = []
    row_pointers = [0]

    for row in dense_matrix:
        for j, value in enumerate(row):
            if value:
                non_zero_values.append(value)
                column_indices.append(j)
        row_pointers.append(len(non_zero_values))

    return non_zero_values, column_indices, row_pointers
