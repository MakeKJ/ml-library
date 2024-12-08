import numpy as np

class BaseOptimizer:
    def __init__(self, model, lr=0.01):
        self.model = model
        self.lr = lr

    def step(self):
        raise NotImplementedError("Subclasses must implement the step method!")


class GDOptimizer:
    """Gradient Descent optimizer."""
    def __init__(self, model, lr=0.01):
        """
        Arguments:
            model:      Model to be optimized (BaseModel)
            lr:         Learning rate (float)
        """
        self.model = model
        self.lr = lr

    def update(self):
        """Get the parameters and gradients of the model and update the parameters."""
        params = self.model.get_parameters()
        gradients = self.model.get_gradients()
        for key in params:
            if key in gradients:
                params[key] -= self.lr * gradients[key]
            else:
                raise ValueError(f"Gradients for {key} not found!")
        self.model.set_parameters(params)