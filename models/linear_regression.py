import numpy as np

class LinearRegression:
    """
    Linear Regression model that is trained using gradient descent.
    """

    def __init__(self):
        self.w = None
        self.b = None

    def train(self, X, Y, epochs=100, lr=0.01):
        """
        Trains the Linear Regression model using gradient descent.

        Arguments:
            X:          Features of the dataset (numpy array)
            Y:          Labels of the dataset (numpy array)
            epochs:     Number of iterations to train the model (int, default=100)
            lr:         Learning rate for the gradient descent (float, default=0.01)
        """
        X = np.asarray(X)
        Y = np.asarray(Y)

        if X.shape[0] != Y.shape[0]:
            raise ValueError("Number of samples in X and Y must match.")

        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(epochs):
            # Forward pass
            Y_pred = np.dot(X, self.w) + self.b

            # Backward pass
            difference = Y_pred - Y
            dw = 2 / n_samples * np.dot(X.T, difference)
            db = 2 / n_samples * np.sum(difference)
            
            # Updating weights and bias
            self.w -= lr * dw
            self.b -= lr * db
                    
    def predict(self, X):
        """
        Predicts the labels of the dataset.

        Arguments:
            X:          Features of the dataset (numpy array)

        Returns:
            Y_pred:     Predicted labels of the dataset (numpy array)
        """
        return np.dot(X, self.w) + self.b