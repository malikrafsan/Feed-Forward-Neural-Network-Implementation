import numpy as np

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)    

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def sign(x):
    return 1 if x > 0 else -1

def linear(x):
    return x
