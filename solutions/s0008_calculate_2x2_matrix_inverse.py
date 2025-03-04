def inverse_2x2(matrix: list[list[float]]) -> list[list[float]]:
    a = matrix[0][0]
    b = matrix[0][1]
    c = matrix[1][0]
    d = matrix[1][1]

    k = 1 / (a * d - b * c)

    matrix[0][0] = round(k * d, 4)
    matrix[0][1] = round(k * -b, 4)
    matrix[1][0] = round(k * -c, 4)
    matrix[1][1] = round(k * a, 4)

    return matrix
