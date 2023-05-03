from model_factory import ModelFactory
import sys

from sklearn import datasets
import pandas as pd

iris = datasets.load_iris()
x = iris.data
y = iris.target


if __name__ == '__main__':
    if (len(sys.argv) > 1):
        model_name = sys.argv[1]
    else:
        model_name = 'model1'

    model_factory = ModelFactory(model_name)
    model = model_factory.create()
    model.summary()

    # data = [[3.0, 1.0],
    #         [1.0, 2.0]]

    data = [[0.0]]

    # target = [
    #     [2.0,  0.3, -1.9],
    #     [1.3, -0.7,  0.1]
    # ]
    target = [
        [0.0, 1.0]
    ]

    stop_reason = model.fit(data, target, max_iterations=1000, learning_rate=0.01, batch_size=1, error_threshold=0.25)
    model.summary()
    print(stop_reason)

    # trained = model(data)
    # print(trained)

    # stopReason = model.fit(x, y, max_iterations=100, learning_rate=0.2)
    # model.summary()
    # print(stopReason)
    # print(y)
    # print(y)
