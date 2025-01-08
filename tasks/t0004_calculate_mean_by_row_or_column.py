def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
    means = []

    if mode == "row":
        for i in range(len(matrix)):
            means.append(sum(matrix[i]) / len(matrix[i]))
    elif mode == "column":
        columns = len(matrix[0])
        means = [0.0] * columns
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                means[j] += matrix[i][j]
        for i in range(columns):
            means[i] /= columns

    return means
