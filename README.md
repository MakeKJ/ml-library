# ml-library

A personal machine learning library project aimed at deepening my understanding of the underlying principles behind various machine learning models. The library provides essential tools for building, training, and experimenting with different machine learning models.

An example usage of the features of the library can be seen in example_mnist_cnn.ipynb file.

## Table of Contents

- [Modules](#modules)
  - [Models](#models)
  - [Neural Networks (NN)](#neural-networks-nn)
  - [Utilities](#utilities)

### Modules

#### Models

The library currently supports various machine learning models:

- **Linear Regression**
  - A Linear Regression model that uses gradient descent to update the parameters.
  - `linear_regression.py`
- **Logistic Regression**
  - A Logistic Regression model that uses stochastic gradient descent to update the parameters.
  - `logistic_regression.py`
- **Primal Support Vector Machine**
  - A Support Vector Machine model implemented in primal representation that uses stochastic gradient descent to update the parameters.
  - `primal_svm.py`
- **Dual Support Vector Machine**
  - A Support Vector Machine model implemented in dual representation that uses stochastic dual coordinate ascent to update the parameters.
  - The model features linear, polynomial, radial basis function (RBF) and Gaussian kernels to allow flexibility in classification tasks.
  - `dual_svm.py`

#### Neural Networks (NN)

This submodule contains components to create and train neural networks.

- **BaseModel**: A base class for neural network models with common methods.
  - `BaseModel.py`

- **Activation Functions**: Common activation functions used in neural networks.
  - **Implemented Activation functions**:
    - [Tanh, Sigmoid, ReLu]
  - `activations.py`

- **Layers**: Different layers used in neural networks.
  - **Implemented Layers**:
    - Linear
    - Convolutional 2D
    - Max pooling 2D
  - `layers.py`

- **Loss Functions**: A collection of loss functions.

  - **Implemented Loss Functions**:
    - [MSE_loss, BCE_loss]
  - `loss_functions.py`

- **Optimizers**: Optimizers used to update model parameters during training.
  - **Implemented Optimizers**:
    - [GDOptimizer]
  - `optimizers.py`

#### Utilities

Utility functions for preprocessing and evaluation.

- **Preprocessing**: Includes data preprocessing utilities.
  - **Implemented Features**:
    - [train_test_split, train_val_test_split]
  - `preprocessing.py`

- **Metrics**: Includes utility functions for evaluation metrics.
  - **Implemented Features**:
    - [mean_squared_error, accuracy, precision, recall, f1_score]
  - `metrics.py`
 
## Contact

For questions and feedback, feel free to reach out:

- Email: mk.junttila@gmail.com
