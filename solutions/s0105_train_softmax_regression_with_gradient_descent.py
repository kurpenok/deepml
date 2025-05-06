import numpy as np


def train_softmaxreg(
    X: np.ndarray, y: np.ndarray, learning_rate: float, iterations: int
) -> tuple[np.ndarray, np.ndarray]:
    n_samples, _ = X.shape
    X_b = np.hstack([np.ones((n_samples, 1)), X])
    D = X_b.shape[1]

    classes = np.unique(y)
    C = len(classes)

    Y_onehot = np.zeros((n_samples, C))
    Y_onehot[np.arange(n_samples), y] = 1

    beta = np.zeros((C, D))

    loss_history = []

    for _ in range(iterations):
        scores = X_b @ beta.T

        max_scores = np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(scores - max_scores)
        softmax_matrix = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        loss = -np.sum(Y_onehot * np.log(softmax_matrix))
        loss_history.append(round(loss, 4))

        gradient = (softmax_matrix - Y_onehot).T @ X_b

        beta -= learning_rate * gradient

    beta_rounded = np.round(beta, 4)
    loss_rounded = np.round(loss_history, 4)

    return beta_rounded, loss_rounded
