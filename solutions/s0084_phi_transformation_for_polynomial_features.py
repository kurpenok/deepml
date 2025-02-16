def phi_transform(data: list[float], degree: int) -> list[list[float]]:
    result = []

    for value in data:
        degree_row = []
        for d in range(degree + 1):
            degree_row.append(value**d)
        result.append(degree_row)

    return result
