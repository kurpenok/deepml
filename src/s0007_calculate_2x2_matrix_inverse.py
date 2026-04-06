def inverse_2x2(matrix: list[list[float]]) -> list[list[float]] | None:
    a, b, c, d = matrix[0][0], matrix[0][1], matrix[1][0], matrix[1][1]

    det = a * d - b * c

    return None if not det else [[d / det, -b / det], [-c / det, a / det]]
