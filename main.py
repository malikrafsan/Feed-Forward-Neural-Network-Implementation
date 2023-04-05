from model_factory import ModelFactory
import sys

if __name__ == '__main__':
    if (len(sys.argv) > 1):
        model_name = sys.argv[1]
    else:
        model_name = 'model1'

    model_factory = ModelFactory(model_name)
    model = model_factory.create()
    model.summary()

    data = [[1.0, 2.0]]
    trained = model(data)
    print(trained)
