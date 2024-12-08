import numpy as np

def train_test_split(X, Y, test_size=0.2, random_seed=None):
    """
    A function that randomly splits a dataset into train and test sets.

    Arguments:
        X:              Features of the dataset (numpy array)
        Y:              Labels of the dataset (numpy array)
        test_size:      The portion of the dataset to be included in the test set (float)
        random_seed:    The possibility to set random seed for reproducability (int)

    Returns:
        X_train:        Features of the training set (numpy array)
        X_test:         Features of the test set (numpy array)
        Y_train:        Labels of the training set (numpy array)
        Y_test:         Labels of the test set (numpy array)
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    num_samples = len(X)
    random_indices = np.random.permutation(num_samples)
    test_size = int(num_samples*test_size)

    test_indices = random_indices[:test_size]
    train_indices = random_indices[test_size:]

    X_train, X_test = X[train_indices], X[test_indices]
    Y_train, Y_test = Y[train_indices], Y[test_indices]

    return X_train, X_test, Y_train, Y_test    
