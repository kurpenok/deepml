def vector_sum(a: list[int | float], b: list[int | float]) -> int | list[int | float]:
    if len(a) != len(b):
        return -1
    return [a_i + b_i for a_i, b_i in zip(a, b)]
