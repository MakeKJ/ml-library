import numpy as np

def accuracy(Y, Y_pred):
    """
    Calculates the accuracy of predictions.

    Arguments:
        Y:          True labels (numpy array)
        Y_pred:     Predicted labels (numpy array)
    
    Returns:
        accuracy:   Accuracy of the predictions (float)
    """
    return np.sum(Y == Y_pred)/len(Y)


def mean_squared_error(Y, Y_pred):
    """
    Calculates the mean squared error loss between the true labels and the predicted labels.

    Arguments:
        Y:          True labels (numpy array)
        Y_pred:     Predicted labels (numpy array)

    Returns:
        loss:       Mean squared error (float)
    """
    return np.mean(np.square(Y - Y_pred))