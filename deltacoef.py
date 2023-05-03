import numpy as np

def linear(o: float):
    return 1

def relu(o: float):
    if(o < 0):
        return 0
    else:
        return 1

def sigmoid(o: float):
    return o*(1-o)

def softmax(o: float):
    # TODO: implement
    return 0

