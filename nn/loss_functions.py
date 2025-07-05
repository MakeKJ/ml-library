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
    

class CategoricalCrossEntropyLoss:
    """Categorical Cross-Entropy loss with integrated Softmax activation for simplicity."""

    def forward(self, logits, y_true):
        """
        Calculates the forward pass of the layer.

        Arguments:
            logits: Raw outputs from the final linear layer of the model. Shape: (batch_size, num_classes) (numpy array)
            y_true: One-hot encoded true labels. Shape: (batch_size, num_classes) (numpy array)

        Returns:
            loss:   The mean cross-entropy loss for the batch.
        """
        self.y_true = y_true
        batch_size = logits.shape[0]

        max_logits = np.max(logits, axis=1, keepdims=True)
        exps = np.exp(logits - max_logits)
        self.y_pred = exps / np.sum(exps, axis=1, keepdims=True)

        # Avoid division by zero in log
        eps = 1e-12
        clipped_y_pred = np.clip(self.y_pred, eps, 1 - eps)

        correct_logprobs = -np.sum(self.y_true * np.log(clipped_y_pred))

        loss = correct_logprobs / batch_size
        return loss

    def backward(self):
        """
        Calculates the backward pass.

        Returns:
            dy:     Gradient of the loss with respect to the logits. Shape: (batch_size, num_classes) (numpy array)
        """
        assert hasattr(self, 'y_pred') and hasattr(self, 'y_true'), "forward() needs to be called first!"
        
        batch_size = self.y_true.shape[0]

        dy = self.y_pred - self.y_true
        
        dy = dy / batch_size
        
        return dy