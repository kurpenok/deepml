def scalar_multiply(
    matrix: list[list[int | float]], scalar: int | float
) -> list[list[int | float]]:
    return [[scalar * e for e in row] for row in matrix]
