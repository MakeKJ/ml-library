import numpy as np

class PrimalSVM:
    """"""
    def __init__(self):
        self.w = None
        self.b = None

    def train(self, X, Y, lr=0.01, epochs=100, C=1, decay=False):
        """"""
        X = np.asarray(X)
        Y = np.asarray(Y)

        if X.shape[0] != Y.shape[0]:
            raise ValueError("Number of samples in X and Y must match.")

        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0
        initial_lr = lr

        # Stochastic gradient descent
        indices = np.arange(n_samples)
        for epoch in range(epochs):
            # Shuffle the data
            np.random.shuffle(indices)
            for i in indices:
                z = Y[i]*np.dot(self.w.T, X[i])

                # Piece-wise defined gradient
                if z < 1:
                    dw = 1/C * self.w - Y[i] * X[i]
                    db = -Y[i]
                else:
                    dw = 1/C * self.w
                    db = 0
                
                # Update the weights
                self.w -= lr * dw
                self.b -= lr * db

            if decay:
                # lr = initial_lr * np.exp(-0.01 * epoch)
                lr = max(initial_lr * np.exp(-0.01 * epoch), 1e-6)

    def predict(self, X):
        """"""
        return np.sign(np.dot(self.w.T, X) + self.b)