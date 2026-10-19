import numpy as np

# One-vs-rest linear SVM trained with full-batch gradient descent.
class MulticlassSVM:
    def __init__(self, learning_rate, lambda_param, n_iters):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters

        self.weights = None
        self.biases = None
        self.classes = None

    def train_binary(self, X, y_bin):
        w = np.zeros(X.shape[1])
        b = 0.0
        for _ in range(self.n_iters):
            scores = X @ w + b
            margins = y_bin * scores  # functional margins
            mask = margins < 1
            # Average subgradient of hinge loss + gradient of lambda*||w||^2/2
            grad_w = self.lambda_param * w
            if np.any(mask):
                grad_w -= (X[mask].T @ y_bin[mask]) / X.shape[0]
                grad_b = -np.sum(y_bin[mask]) / X.shape[0]
            else:
                grad_b = 0.0
            # Gradient descent update
            w -= self.learning_rate * grad_w
            b -= self.learning_rate * grad_b
        return w, b

    def fit(self, X, y):
        self.classes = np.unique(y)
        n_features = X.shape[1]
        n_classes = len(self.classes)

        self.weights = np.zeros((n_classes, n_features))
        self.biases = np.zeros(n_classes)

        for i, c in enumerate(self.classes):
            w, b = self.train_binary(X, (y == c)*2-1)
            self.weights[i] = w
            self.biases[i] = b

        return self

    def predict(self, X):
        assert self.weights is not None
        assert self.biases is not None
        assert self.classes is not None

        scores = X @ self.weights.T + self.biases
        idx = np.argmax(scores, axis=1)
        return self.classes[idx]
