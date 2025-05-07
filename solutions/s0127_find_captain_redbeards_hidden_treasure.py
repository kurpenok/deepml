def find_treasure(start_x: float) -> float:
    learning_rate = 0.001
    tolerance = 1e-8
    max_iterations = 1000000

    x = start_x

    for _ in range(max_iterations):
        derivative = 4 * x**3 - 9 * x**2
        step = learning_rate * derivative
        new_x = x - step
        if abs(step) < tolerance:
            break
        x = new_x

    return x
