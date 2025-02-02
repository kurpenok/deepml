def compressed_row_sparse_matrix(
    dense_matrix: list[list[int]],
) -> tuple[list[int], list[int], list[int]]:
    elements = []
    column_indices = []
    row_pointers = [0]

    for row in dense_matrix:
        for j, element in enumerate(row):
            if element:
                elements.append(element)
                column_indices.append(j)
        row_pointers.append(len(elements))

    return elements, column_indices, row_pointers
