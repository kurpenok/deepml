def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
    if mode == "column":
        return [sum(column) / len(column) for column in zip(*matrix)]
    elif mode == "row":
        return [sum(row) / len(row) for row in matrix]
    else:
        return []
