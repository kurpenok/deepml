import math


def calculate_eigenvalues(matrix: list[list[int | float]]) -> list[float]:
    a = matrix[0][0]
    b = matrix[0][1]
    c = matrix[1][0]
    d = matrix[1][1]

    trace = a + d
    det = a * d - b * c
    discr = trace**2 - 4 * det

    return [(trace + math.sqrt(discr)) / 2, (trace - math.sqrt(discr)) / 2]
