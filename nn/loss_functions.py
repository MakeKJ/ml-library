import numpy as np

class MSE_loss:
    """Mean Squared Error loss."""

    def forward(self, y_pred, y):
        self.difference = y_pred - y
        return np.sum(np.square(self.difference)) / self.difference.size
    
    def backward(self):
        assert hasattr(self, 'difference'), "forward() needs to be called first!"
        dy = 2 / self.difference.size * self.difference
        return dy


class BCE_loss:
    """Binary Cross-Entropy loss."""

    def forward(self, y_pred, y):
        self.y = y
        self.y_pred = y_pred

        # Avoid division by zero
        eps = 1e-12
        self.y_pred = np.clip(self.y_pred, eps, 1 - eps)

        loss = -np.mean(y * np.log(self.y_pred) + (1 - y) * np.log(1 - self.y_pred))
        return loss
    
    def backward(self):
        assert hasattr(self, 'y') and hasattr(self, 'y_pred'), "forward() needs to be called first!"
        dy = - (self.y / (self.y_pred) - (1 - self.y) / (1 - self.y_pred)) / self.y.size
        return dy