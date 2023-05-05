from model_factory import ModelFactory
from json_parser import JsonParser
from model_types import ModelConfig
from ann import StopReason, Model
import sys

from sklearn import datasets
import pandas as pd
import argparse

iris = datasets.load_iris()
x = iris.data
y = iris.target


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='main.py', description='Feed Forward Neural Network')
    parser.add_argument('json_path', type=str, help='path to json test file')
    parser.add_argument('-s', '--save', type=str, help='path to save model')
    parser.add_argument('-l', '--load', type=str, help='path to load model. If specified, json_path will only be used to load the input data')
    args = parser.parse_args()
    print(args)

    model_config: ModelConfig = JsonParser().parse_model_config(args.json_path)
    model_factory = ModelFactory()
    if (args.load is not None):
        model = model_factory.load(args.load)
    else:
        model = model_factory.build(model_config)

    model.summary()

    data = model_config['case']['input']
    target = model_config['case']['target']

    stop_reason = model.fit(data, target)
    print("===================================================")
    model.summary()
    print("Expected final weights:",
          model_config['expect'].get("final_weights"))

    print("===================================================")
    if (stop_reason == StopReason.MAX_ITERATIONS):
        print('Stop reason: max_iterations')
    elif (stop_reason == StopReason.ERROR_THRESHOLD):
        print('Stop reason: error_threshold')

    print("Expected stop reason: ", model_config['expect']['stopped_by'])

    print("===================================================")
    model.draw()

    if (args.save is not None):
        model.save(args.save)

    # trained = model(data)
    # print(trained)

    # stopReason = model.fit(x, y, max_iterations=100, learning_rate=0.2)
    # model.summary()
    # print(stopReason)
    # print(y)
    # print(y)
