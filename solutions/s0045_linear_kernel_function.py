def kernel_function(x_1: list[float], x_2: list[float]) -> float:
    return sum([x * y for x, y in zip(x_1, x_2)])
