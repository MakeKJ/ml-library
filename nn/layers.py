import numpy as np

class Linear:
    """
    Linear layer in a neural network.
    """

    def __init__(self, x_dim, y_dim):
        """
        Initialize the layer.

        Arguments:
            x_dim: input dimension (int)
            y_dim: output dimension (int)
        """
        # Initialize the weights
        # bound = 3 / np.sqrt(x_dim)
        # self.w = np.random.uniform(-bound, bound, (y_dim, x_dim))
        # bound = 1 / np.sqrt(x_dim)
        # self.b = np.random.uniform(-bound, bound, y_dim)

        bound = np.sqrt(1 / x_dim)
        self.w = np.random.uniform(-bound, bound, (y_dim, x_dim))
        self.b = np.zeros(y_dim)  # Biases initialized to zero

        self.dw = None
        self.db = None

    def forward(self, x):
        """
        Forward pass.

        Arguments:
            x: input data (numpy array)

        Returns:
            y: output data (numpy array)
        """ 
        self.x = x
        y = np.dot(x, self.w.T) + self.b
        return y

    def backward(self, dy):
        """
        Backward pass.

        Arguments:
            dy: gradient of the loss with respect to the output (numpy array)

        Returns:
            dx: gradient of the loss with respect to the input (numpy array)
        """
        self.dw = np.dot(dy.T, self.x)
        self.db = np.sum(dy, axis=0)
        dx = np.dot(dy, self.w)
        return dx