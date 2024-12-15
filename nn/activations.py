import numpy as np

class Tanh:
    """
    Tanh activation function.
    """
    def forward(self, x):
        """
        Arguments:
            x:      Inputs (n_features,) (numpy array)

        Returns:
            y:     Output of the Tanh function (n_features,) (numpy array)
        """
        self.x = x
        self.y = np.tanh(x)
        y = np.tanh(x)
        return y

    def backward(self, dy):
        """
        Arguments:
            dy of shape (n_features,): Gradient of the loss with respect to y.

        Returns:
            dx of shape (n_features,): Gradient of the loss with respect to x.
        """
        assert hasattr(self, 'x'), "Need to call forward() first."
        dx = dy * (1 - self.y**2)
        return dx
    

class Sigmoid:
    """
    Sigmoid activation function.
    """
    def forward(self, x):
        """
        Arguments:
            x:      Inputs (n_features,) (numpy array)

        Returns:
            y:     Output of the Sigmoid function (n_features,) (numpy array)
        """
        self.x = x
        # A treshold to avoid overflow in exp(x)
        threshold = 700
        x_clipped = np.clip(x, -threshold, threshold)
        
        self.y = 1 / (1 + np.exp(-x_clipped))
        return self.y

    def backward(self, dy):
        """
        Arguments:
            dy:     Gradient of the loss with respect to y (n_features,) (numpy array)

        Returns:
            dx:     Gradient of the loss with respect to x (n_features,) (numpy array)
        """
        assert hasattr(self, 'x'), "Need to call forward() first."
        dx = dy * self.y * (1 - self.y)
        return dx
    

class ReLu:
    """
    ReLu activation function.
    """
    def forward(self, x):
        """
        Arguments:
            x:      Inputs (n_features,) (numpy array)

        Returns:
            y:     Output of the ReLu function (n_features,) (numpy array)
        """
        self.x = x
        self.y = np.maximum(0, x)
        return self.y

    def backward(self, dy):
        """
        Arguments:
            dy:     Gradient of the loss with respect to y (n_features,) (numpy array)

        Returns:
            dx:     Gradient of the loss with respect to x (n_features,) (numpy array)
        """
        assert hasattr(self, 'x'), "Need to call forward() first."
        dx = dy * (self.x > 0)
        return dx