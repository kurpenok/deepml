from collections import Counter


def performance_metrics(
    actual: list[int], predicted: list[int]
) -> tuple[list, float, float, float, float]:
    if len(actual) != len(predicted):
        raise ValueError("Lists must be of the same length")

    data = list(zip(actual, predicted))
    counts = Counter(tuple(pair) for pair in data)

    tp, fn, fp, tn = counts[(1, 1)], counts[(1, 0)], counts[(0, 1)], counts[(0, 0)]
    confusion_matrix = [[tp, fn], [fp, tn]]

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f_1 = 2 * precision * recall / (precision + recall)

    negative_predictive = tn / (tn + fn)
    specificity = tn / (tn + fp)

    return (
        confusion_matrix,
        round(accuracy, 3),
        round(f_1, 3),
        round(specificity, 3),
        round(negative_predictive, 3),
    )
