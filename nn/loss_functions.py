import numpy as np

class MSE_loss:

    def forward(self, y, y_pred):
        self.difference = y_pred - y
        # return np.mean(np.square(self.difference))
        return np.sum(np.square(self.difference)) / self.difference.size
    
    def backward(self):
        assert hasattr(self, 'difference'), "forward() needs to be called first!"
        dy = 2 / self.difference.size * self.difference
        return dy
    
