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


def softmax(cur_out: float, other_out: float, cur_idx: int, other_idx: int):
    print("cur_out", cur_out, other_out, cur_idx, other_idx)
    return other_out if (cur_idx != other_idx) else (-1 * (1 - cur_out))
