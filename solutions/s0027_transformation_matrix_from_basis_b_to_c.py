def inverse_matrix(matrix: list[list[float]]) -> list[list[float]]:
    n = len(matrix)

    extended_matrix = []
    for i in range(n):
        row = matrix[i] + [0] * n
        row[n + i] = 1
        extended_matrix.append(row)

    for i in range(n):
        if extended_matrix[i][i] == 0:
            for j in range(i + 1, n):
                if extended_matrix[j][i] != 0:
                    extended_matrix[i], extended_matrix[j] = (
                        extended_matrix[j],
                        extended_matrix[i],
                    )
                    break

        pivot = extended_matrix[i][i]
        extended_matrix[i] = [x / pivot for x in extended_matrix[i]]

        for j in range(n):
            if i != j:
                multiplier = extended_matrix[j][i]
                extended_matrix[j] = [
                    extended_matrix[j][k] - multiplier * extended_matrix[i][k]
                    for k in range(2 * n)
                ]

    inverse = []
    for i in range(n):
        inverse_row = extended_matrix[i][n:]
        inverse.append(inverse_row)

    return inverse


def dot(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    m = len(A)
    n = len(B[0])
    p = len(B)

    result = [[0.0 for _ in range(n)] for _ in range(m)]

    for i in range(m):
        for j in range(n):
            for k in range(p):
                result[i][j] += A[i][k] * B[k][j]

    return result


def transform_basis(B: list[list[float]], C: list[list[float]]) -> list[list[float]]:
    return [[round(cell, 4) for cell in row] for row in dot(inverse_matrix(C), B)]
