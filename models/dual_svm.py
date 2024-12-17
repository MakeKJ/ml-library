import numpy as np

# Implementing commonly used kernel functions
def linear_kernel(X, Y=None):
    if Y is None:
        Y = X
    return np.dot(X, Y.T)

def polynomial_kernel(X, Y=None, degree=3, constant=1):
    if degree <= 0:
        raise ValueError("Degree must be a positive integer.")

    if Y is None:
        Y = X
    return (constant + np.dot(X, Y.T)) ** degree

def rbf_kernel(X, Y=None, gamma=1):  # Radial Basis Function
    if Y is None:
        Y = X
    # X[:, None] - Y[None, :] used for broadcasting to calculate the pairwise distances
    K = np.exp(-gamma * np.sum((X[:, None] - Y[None, :]) ** 2, axis=-1))
    return K

def sigmoid_kernel(X, Y=None, gamma=1, constant=0):
    if Y is None:
        Y = X
    return np.tanh(gamma * np.dot(X, Y.T) + constant)

class DualSVM:
    """
    Support Vector Machine (SVM) model using the dual formulation that is trained using stochastic dual coordinate ascent.
    """

    def __init__(self):
        self.alpha = None
        self.b = None
        self.K = None
        self.kernel = None
        self.x_support = None
        self.y_support = None
        self.alpha_support = None
    
    def train(self, X, Y, epochs=100, C=1, kernel=linear_kernel, **kernel_params):
        """
        Trains the Dual SVM model using stochastic dual coordinate ascent.
        
        Arguments:
            X:              Features of the dataset (numpy array)
            Y:              Labels of the dataset (numpy array)
            epochs:         Number of iterations to train the model (int, default=100)
            C:              Regularization parameter (float, default=1)
            kernel:         Kernel function to use (function, default=linear_kernel)
            kernel_params:  Additional parameters for the kernel function (see the kernel functions for details)
        """
        X = np.asarray(X)
        Y = np.asarray(Y)

        if X.shape[0] != Y.shape[0]:
            raise ValueError("Number of samples in X and Y must match.")

        n_samples, n_features = X.shape
        self.alpha = np.zeros(n_samples)
        self.kernel = kernel
        self.K = kernel(X, X, **kernel_params)

        # Update the dual variables
        for epoch in range(epochs):
            for i in range(n_samples):
                # The difference to the optimum
                d_alpha = (1 - Y[i] * np.sum(self.alpha * Y * self.K[i])) / (self.K[i, i])

                # Update the dual variable
                alpha_new = self.alpha[i] + d_alpha
                self.alpha[i] = np.minimum(np.maximum(0, alpha_new), C)  # To satisfy constraints
        
        # Compute the support vectors
        support_vectors_indices = self.alpha > 0
        self.alpha_support = self.alpha[support_vectors_indices]
        self.x_support = X[support_vectors_indices]
        self.y_support = Y[support_vectors_indices]

        K_support = self.K[support_vectors_indices][:, support_vectors_indices]
        self.b = np.mean(self.y_support - np.sum(self.alpha_support * self.y_support * K_support, axis=1))

    def predict(self, X):
        """
        Predicts the labels of the dataset.
        
        Arguments:
            X:          Features of the dataset (numpy array)
        """
        X = np.atleast_2d(X)  # Converts 1D input to shape (1, n_features)
        kernels = self.kernel(X, self.x_support)
        return np.sign(np.sum(  self.alpha_support * self.y_support * kernels, axis=1) + self.b)