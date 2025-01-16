import math


def sigmoid(z: float) -> float:
    return 1 / (1 + math.exp(-z))


def mean_squared_error(y_true: list[float], y_pred: list[float]) -> float:
    return sum([(y_hat - y) ** 2 for y, y_hat in zip(y_true, y_pred)]) / len(y_true)


def single_neuron_model(
    features: list[list[float]],
    labels: list[float],
    weights: list[float],
    bias: float,
) -> tuple[list[float], float]:
    probabilities = []

    for feature in features:
        theta = sum([f * w for f, w in zip(feature, weights)]) + bias
        probability = sigmoid(theta)
        probabilities.append(probability)

    mse = round(mean_squared_error(labels, probabilities), 4)
    probabilities = [round(p, 4) for p in probabilities]

    return probabilities, mse
