from json_reader import JsonReader
from ann import Model, Layer

class ModelFactory:
    def __init__(self, model_name):
        self.model_name = model_name
        self.json_reader = JsonReader(f'{model_name}.json')
        self.model = None

    def create(self) -> Model:
        self.json_reader.read()
        self.model = Model()
        for i in range(self.json_reader.length()):
            layer = self.json_reader.get(i)
            self.model.add(Layer(
                layer['neurons'],
                input_shape=layer['neurons'] + 1,
                activation=layer['activation'],
                weights=layer['weights'],
                bias=layer['bias'],
            ))
        return self.model