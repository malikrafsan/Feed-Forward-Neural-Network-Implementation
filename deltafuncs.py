import numpy as np


def linear(expected: float, value: float):
    return -(expected-value)


def relu(expected: float, value: float):
    if (value < 0):
        return 0
    else:
        return -(expected - value)


def sigmoid(expected: float, value: float):
    return -value * (1 - value) * (expected - value)


def softmax(expected: float, value: float):
    # TODO: implement softmax
    raise NotImplementedError
