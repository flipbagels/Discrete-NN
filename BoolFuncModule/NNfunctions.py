import numpy as np

def relu(x):
    return x * (x > 0)

def sign(x):
    return 1 * (x > 0)

def forward_pass(parameters, x):
    if not isinstance(parameters, tuple):
        raise TypeError('parameters must be tuple.')
    W1, b1, W2, b2 = parameters
    x = W1@x + b1
    x = relu(x)
    x = W2@x + b2
    x = sign(x)
    return x
