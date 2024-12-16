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
    # Ensure int arrays
    Y = np.array(Y, dtype=int)
    Y_pred = np.array(Y_pred, dtype=int)

    return np.mean(Y == Y_pred)


def precision(Y, Y_pred):
    """
    Calculates the precision of predictions.

    Arguments:
        Y:          True labels (numpy array)
        Y_pred:     Predicted labels (numpy array)
    
    Returns:
        precision:  Precision of the predictions (float)
    """
    # Ensure int arrays
    Y = np.array(Y, dtype=int)
    Y_pred = np.array(Y_pred, dtype=int)

    true_positives = np.sum((Y == 1) & (Y_pred == 1))
    false_positives = np.sum((Y == -1) & (Y_pred == 1))
    return true_positives / (true_positives + false_positives)


def recall(Y, Y_pred):
    """
    Calculates the recall of predictions.

    Arguments:
        Y:          True labels (numpy array)
        Y_pred:     Predicted labels (numpy array)
    
    Returns:
        recall:     Recall of the predictions (float)
    """
    # Ensure int arrays
    Y = np.array(Y, dtype=int)
    Y_pred = np.array(Y_pred, dtype=int)

    true_positives = np.sum((Y == 1) & (Y_pred == 1))
    false_negatives = np.sum((Y == 1) & (Y_pred == -1))
    return true_positives / (true_positives + false_negatives)


def f1_score(Y, Y_pred):
    """
    Calculates the F1 score of predictions.

    Arguments:
        Y:          True labels (numpy array)
        Y_pred:     Predicted labels (numpy array)
    
    Returns:
        f1_score:   F1 score of the predictions (float)
    """
    # Ensure int arrays
    Y = np.array(Y, dtype=int)
    Y_pred = np.array(Y_pred, dtype=int)

    true_positives = np.sum((Y == 1) & (Y_pred == 1))
    false_positives = np.sum((Y == -1) & (Y_pred == 1))
    false_negatives = np.sum((Y == 1) & (Y_pred == -1))
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    return 2 * (precision * recall) / (precision + recall)


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