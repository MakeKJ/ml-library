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
        # He initialization
        std = np.sqrt(2 / x_dim)
        self.w = np.random.normal(0, std, (y_dim, x_dim))
        self.b = np.zeros(y_dim)

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
        
        # Handle case where x has more than 2 dimensions
        if len(x.shape) > 2:
            x_reshaped = x.reshape(-1, x.shape[-1])
        else:
            x_reshaped = x
        
        y = np.dot(x_reshaped, self.w.T) + self.b
        
        # Reshape back to the original batch shape if necessary
        if len(x.shape) > 2:
            y = y.reshape(x.shape[0], -1, self.w.shape[0])
        
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