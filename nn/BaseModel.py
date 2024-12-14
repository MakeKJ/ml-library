import numpy as np

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
            if f"layer_{i}_weights" in new_params:
                layer.w = new_params[f"layer_{i}_weights"]
            else:
                raise ValueError(f"Layer {i} does not have weights!")
            if f"layer_{i}_biases" in new_params:
                layer.b = new_params[f"layer_{i}_biases"]

    def get_parameters(self):
        """Extract all trainable parameters from layers."""
        params = {}
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'w') and hasattr(layer, 'b'):
                params[f"layer_{i}_weights"] = layer.w
                params[f"layer_{i}_biases"] = layer.b
        return params
    
    def get_gradients(self):
        """Extract gradients from all layers."""
        gradients = {}
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'dw') and hasattr(layer, 'db'):
                gradients[f"layer_{i}_weights"] = layer.dw
                gradients[f"layer_{i}_biases"] = layer.db
        return gradients

    def forward(self, x):
        """Define the forward pass. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement forward!")

    def backward(self, dy):
        """Define the backward pass. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement backward!")