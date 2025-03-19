def OSA(source: str, target: str) -> int:
    osa_matrix = [[0] * (len(target) + 1) for _ in range(len(source) + 1)]

    for j in range(1, len(target) + 1):
        osa_matrix[0][j] = j

    for i in range(1, len(source) + 1):
        osa_matrix[i][0] = i

    for i in range(1, len(source) + 1):
        for j in range(1, len(target) + 1):
            osa_matrix[i][j] = min(
                osa_matrix[i - 1][j] + 1,
                osa_matrix[i][j - 1] + 1,
                osa_matrix[i - 1][j - 1] + (1 if source[i - 1] != target[j - 1] else 0),
            )
            if (
                i > 1
                and j > 1
                and source[i - 1] == target[j - 2]
                and source[i - 2] == target[j - 1]
            ):
                osa_matrix[i][j] = min(osa_matrix[i][j], osa_matrix[i - 2][j - 2] + 1)

    return osa_matrix[-1][-1]
