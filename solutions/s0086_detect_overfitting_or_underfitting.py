def model_fit_quality(training_accuracy: float, test_accuracy: float) -> int:
    if training_accuracy - test_accuracy > 0.2:
        return 1
    elif training_accuracy < 0.7 and test_accuracy < 0.7:
        return -1
    else:
        return 0
