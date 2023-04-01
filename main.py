from ann import Model, Layer
import numpy as np

if __name__ == '__main__':
    model = Model()
    model.add(Layer(
        2,
        input_shape=3,
        activation='sigmoid',
        weights=[[0.15, 0.2, 0.35], [0.25, 0.3, 0.35]],
        bias=1,
    ))
    model.add(Layer(
        2,
        input_shape=3,
        activation='sigmoid',
        weights=[[0.4, 0.45, 0.6], [0.5, 0.55, 0.6]],
        bias=1,
    ))
    model.summary()

    data = [[0.05, 0.1]]
    trained = model(data)
    print(trained)
