import numpy as np

def sigmoid(x):
    """
    Computes the sigmoid activation function.

    Arguments:
        x:      Input to the activation function

    Returns:
        sigmoid:    Output of the sigmoid function
    """
    return 1 / (1 + np.exp(-x))


def relu(x):
    """
    Computes the ReLU activation function.

    Arguments:
        x:      Input to the activation function

    Returns:
        relu:    Output of the ReLU function
    """
    return np.maximum(0, x)