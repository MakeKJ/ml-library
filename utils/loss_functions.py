import numpy as np

def mean_squared_error_loss(Y, Y_pred):
    """
    Calculates the mean squared error loss between the true labels and the predicted labels.

    Arguments:
        Y:          True labels (numpy array)
        Y_pred:     Predicted labels (numpy array)
    
    Returns:
        loss:       Mean squared error (float)
    """
    return np.mean(np.square(Y - Y_pred))