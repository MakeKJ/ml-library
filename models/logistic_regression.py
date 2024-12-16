import numpy as np

class LogisticRegression:
    """
    Logistic Regression classifier that is trained using stochastic gradient descent.
    """

    def __init__(self):
        """Initializes the Logistic Regression model."""
        self.w = None

    def train(self, X, Y, lr=0.01, epochs=100, decay=False):
        """
        Trains the logistic regression model using stochastic gradient descent.

        Arguments:
            X:              Features of the dataset (numpy array)
            Y:              Labels of the dataset (numpy array)
            max_iterations: Maximum number of iterations to train the model (int)
            lr:             Learning rate for the model (float)
            decay:          Whether to use a decaying learning rate (bool)
        """
        X = np.asarray(X)
        Y = np.asarray(Y)

        n_samples, feature_dim = X.shape
        self.w = np.zeros(feature_dim)
        initial_lr = lr

        if X.shape[0] != Y.shape[0]:
            raise ValueError("Number of samples in X and Y must match.")

        indices = np.arange(n_samples)
        for epoch in range(epochs):
            # Shuffle the data
            np.random.shuffle(indices)
            
            for i in indices:
                # Compute the gradient
                z = Y[i] * np.dot(X[i], self.w)
                
                # For numerical stability
                treshold = 700
                z = np.clip(z, -treshold, treshold)
                sigmoid = np.where(
                    z >= 0,
                    1 / (1 + np.exp(-z)),
                    np.exp(z) / (1 + np.exp(z))
                )
                dw = -Y[i] * X[i] * sigmoid

                # Update the weights
                self.w -= lr * dw

            if decay:
                lr = initial_lr * np.exp(-0.01 * epoch)

    def predict(self, X):
        """
        Predicts the labels of the dataset.

        Arguments:
            X:          Features of the dataset (numpy array)

        Returns:
            Y_pred:     Predicted labels of the dataset (numpy array)
        """
        X = np.asarray(X)
        z = np.dot(X, self.w)

        # For numerical stability
        treshold = 700
        z = np.clip(z, -treshold, treshold)
        sigmoid = np.where(
            z >= 0,
            1 / (1 + np.exp(-z)),
            np.exp(z) / (1 + np.exp(z))
        )
        return np.where(sigmoid >= 0.5, 1, -1)