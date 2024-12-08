# Epic_ML_library

A personal machine learning library project aimed at deepening my understanding of the underlying principles behind various machine learning models. The library provides essential tools for building, training, and experimenting with different machine learning models.

## Table of Contents

- [Modules](#modules)
  - [Models](#models)
  - [Neural Networks (NN)](#neural-networks-nn)
  - [Utilities](#utilities)

### Modules

#### Models

The library currently supports various machine learning models:

- **Linear Regression**: A simple implementation of linear regression that uses gradient descent to update the parameters.
  - `linear_regression.py`

#### Neural Networks (NN)

This submodule contains components to create and train neural networks.

- **BaseModel**: A base class for neural network models with common methods.
  - `BaseModel.py`

- **Activation Functions**: Common activation functions used in neural networks.
  - `activations.py`
  - **Implemented Activation functions**:
    - [Tanh, Sigmoid, ReLu]

- **Layers**: Different layers used in neural networks.
  - `layers.py`
  - **Implemented Layers**:
    - [Linear]

- **Loss Functions**: A collection of loss functions.
  - `loss_functions.py`
  - **Implemented Loss Functions**:
    - [MSE_loss]

- **Optimizers**: Optimizers used to update model parameters during training.
  - `optimizers.py`
  - **Implemented Features**:
    - [GDOptimizer]

#### Utilities

Utility functions for preprocessing and evaluation.

- **Preprocessing**: Includes data preprocessing utilities.
  - `preprocessing.py`
  - **Implemented Features**:
    - [train_test_split]

- **Metrics**: Includes utility functions for evaluation metrics.
  - `metrics.py`
  - **Implemented Features**:
    - [accuracy, mean_squared_error]
