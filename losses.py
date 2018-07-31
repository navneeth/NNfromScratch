import numpy as np

# Define the non-linear activation function
# Logistic sigmoid
def logistic(z):
    return 1 / (1 + np.exp(-z))

# Derivative of logistic function
def logistic_deriv(y):
    return np.multiply(y, (1 - y))

def softmax(z):
    return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)
