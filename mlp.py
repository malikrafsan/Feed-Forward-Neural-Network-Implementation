from sklearn.metrics import classification_report, confusion_matrix
from sklearn.datasets import load_iris
from sklearn.neural_network import MLPRegressor, MLPClassifier
from model_factory import ModelFactory
from json_parser import JsonParser
from model_types import ModelConfig
from ann import StopReason, Model, Layer
import sys
import random

from sklearn import datasets
import pandas as pd
import argparse

iris = datasets.load_iris()
x = iris.data
y = iris.target


def mlpRegressor(X, Y):
    res = [[] for i in range(len(y))]
    for i in range(len(Y[0])):
        mlp = MLPRegressor(activation="identity",
                           random_state=42, hidden_layer_sizes=(2,))
        curY = [Y[j][i] for j in range(len(Y))]

        mlp.fit(X, curY)

        res1D = mlp.predict(X)
        for i in range(len(Y)):
            res[i].append(res1D[i])
    return res


def mlpError(Y_real, Y_pred):
    res = 0
    for i in range(len(Y_real)):
        for j in range(len(Y_real[0])):
            res += (Y_real[i][j]-Y_pred[i][j])**2
    return res


def main():
    data = load_iris()

    X_iris = data.data
    y_iris = data.target

    model_iris_sklearn = MLPClassifier(hidden_layer_sizes=(
        4,), activation="relu", learning_rate_init=0.0025, batch_size=8, max_iter=1000, verbose=True, solver="sgd", momentum=0)

    model_iris_sklearn.fit(X_iris, y_iris)

    prediction = model_iris_sklearn.predict(X_iris) 
    
    report_sklearn = classification_report(y_iris, prediction)


    mlp_scratch = Model(learning_rate=0.1, max_iterations=1000,
                        batch_size=4, error_threshold=3)

    mlp_scratch.add(
        Layer(input_shape=4, neurons=4, 
              activation='sigmoid',
              weights=[[(random.random() - 0.5) for _ in range(5)] for _ in range(4)]))
    mlp_scratch.add(
        Layer(neurons=6,
              activation='sigmoid',
              weights=[[(random.random() - 0.5) for _ in range(5)] for _ in range(6)]))
    mlp_scratch.add(
        Layer(neurons=3, 
              activation='linear',
              weights=[[(random.random() - 0.5) for _ in range(7)] for _ in range(3)]))

    y = [[1 if t == i else 0 for i in range(3)] for t in data.target]
    stop_reason = mlp_scratch.fit(data.data, y)
    print(stop_reason)

    res = mlp_scratch(data.data)
    idx_max = [i.index(max(i)) for i in res]
    
    report_model = classification_report(data.target, idx_max)

    print("report_sklearn")
    print(report_sklearn)

    print("report_model")
    print(report_model)


if __name__ == '__main__':
    main()
