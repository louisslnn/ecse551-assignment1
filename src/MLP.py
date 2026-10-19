import numpy as np

def relu(z):
    return np.maximum(0, z)

def relu_deriv(z):
    return z > 0

def softmax(z):
    # Subtract max for numerical stability
    exp_z = np.exp(z - np.max(z, axis=0))
    return exp_z / np.sum(exp_z, axis=0)

# 2-layer MLP (input -> ReLU hidden -> softmax) using full-batch gradient descent
class MLP:
    def __init__(self, input_size, hidden_size, output_size, lr):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = lr
        # Randomly initialize with small values to break symmetry
        rng = np.random.default_rng(42)
        self.W1 = rng.normal(0, 0.001, size=(hidden_size, input_size))
        self.b1 = np.zeros((hidden_size, 1))
        self.W2 = rng.normal(0, 0.001, size=(output_size, hidden_size))
        self.b2 = np.zeros((output_size, 1))

    def fit(self, X, y, epochs):
        a0 = X.T
        y = np.eye(self.output_size)[y].T  # One-hot encoding

        for _ in range(epochs):
            # Forward pass
            z1 = self.W1 @ a0 + self.b1
            a1 = relu(z1)
            z2 = self.W2 @ a1
            a2 = softmax(z2)
            # Backpropagation
            grad_z2 = a2 - y
            grad_W2 = grad_z2 @ a1.T / X.shape[0]
            grad_b2 = np.mean(grad_z2, axis=1, keepdims=True)
            grad_a1 = self.W2.T @ grad_z2
            grad_z1 = grad_a1 * relu_deriv(z1)
            grad_W1 = grad_z1 @ a0.T / X.shape[0]
            grad_b1 = np.mean(grad_z1, axis=1, keepdims=True)

            self.W2 -= self.lr * grad_W2
            self.b2 -= self.lr * grad_b2
            self.W1 -= self.lr * grad_W1
            self.b1 -= self.lr * grad_b1

        return self

    def predict(self, X):
        z1 = self.W1 @ X.T + self.b1
        z2 = self.W2 @ relu(z1) + self.b2
        return np.argmax(z2, axis=0).T
