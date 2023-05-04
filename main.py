from model_factory import ModelFactory
from json_parser import JsonParser
from model_types import ModelConfig
from ann import StopReason
import sys

from sklearn import datasets
import pandas as pd

iris = datasets.load_iris()
x = iris.data
y = iris.target


if __name__ == '__main__':
    if (len(sys.argv) > 1):
        json_path = sys.argv[1]
    else:
        json_path = 'model1.json'

    model_config: ModelConfig = JsonParser().parse_model_config(json_path)
    model = ModelFactory().build(model_config)

    model.summary()

    data = model_config['case']['input']
    target = model_config['case']['target']

    stop_reason = model.fit(data, target)
    model.summary()
    print("Expected final weights:", model_config['expect']["final_weights"])

    if (stop_reason == StopReason.MAX_ITERATIONS):
        print('Stop reason: max_iterations')
    elif (stop_reason == StopReason.CONVERGENCE):
        print('Stop reason: convergence')
    
    print("Expected stop reason: ", model_config['expect']['stopped_by'])
    # trained = model(data)
    # print(trained)

    # stopReason = model.fit(x, y, max_iterations=100, learning_rate=0.2)
    # model.summary()
    # print(stopReason)
    # print(y)
    # print(y)
