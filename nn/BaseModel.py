import numpy as np
from nn.layers import *

class BaseModel:
    """
    Base class for all neural networks.
    """
    def __init__(self, **kwargs):
        """Initialize the model."""
        self.params = kwargs
        self.layers = []

    def add_layer(self, layer):
        """Add new layer to the model."""
        self.layers.append(layer)

    def set_parameters(self, new_params):
        """Update parameters of the model."""
        for i, layer in enumerate(self.layers):

            # Handle Linear layers
            if isinstance(layer, Linear):
                layer.w = new_params[f"layer_{i}_weights"]
                layer.b = new_params[f"layer_{i}_biases"]
            # Handle Conv2D layers
            elif isinstance(layer, Conv2D):
                layer.K = new_params[f"layer_{i}_weights"]
                layer.b = new_params[f"layer_{i}_biases"]

    def get_parameters(self):
        """Extract all trainable parameters from layers."""
        params = {}
        for i, layer in enumerate(self.layers):

            # Handle Linear layers
            if isinstance(layer, Linear):
                params[f"layer_{i}_weights"] = layer.w
                params[f"layer_{i}_biases"] = layer.b
            elif isinstance(layer, Conv2D):
                params[f"layer_{i}_weights"] = layer.K
                params[f"layer_{i}_biases"] = layer.b   

        return params    

    def get_gradients(self):
        """Extract gradients from all layers."""
        gradients = {}
        for i, layer in enumerate(self.layers):
            if isinstance(layer, Linear):
                gradients[f"layer_{i}_weights"] = layer.dw
                gradients[f"layer_{i}_biases"] = layer.db
            elif isinstance(layer, Conv2D):
                gradients[f"layer_{i}_weights"] = layer.dw
                gradients[f"layer_{i}_biases"] = layer.db

        return gradients

    def forward(self, x):
        """Define the forward pass. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement forward!")

    def backward(self, dy):
        """Define the backward pass. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement backward!")