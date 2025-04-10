def min_max(x: list[int]) -> list[float]:
    normalized_x: list[float] = [float(x_i) for x_i in x]

    min_x = min(x)
    max_x = max(x)

    if min_x == max_x:
        return [0.0] * len(x)

    for i in range(len(x)):
        normalized_x[i] = round((x[i] - min_x) / (max_x - min_x), 4)

    return normalized_x
