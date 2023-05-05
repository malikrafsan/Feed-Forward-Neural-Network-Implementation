from json_reader import JsonReader
from ann import Model, Layer
from model_types import ModelConfig
import pickle

class ModelFactory:
    def build(self, model_config: ModelConfig) -> Model:
        case = model_config["case"]
        learning_parameters = case["learning_parameters"]
        model = Model(
            learning_rate=learning_parameters["learning_rate"],
            batch_size=learning_parameters["batch_size"],
            max_iterations=learning_parameters["max_iteration"],
            error_threshold=learning_parameters["error_threshold"]
        )
        for i in range(len(case["model"]["layers"])):
            layer = case["model"]["layers"][i]
            weights = case["initial_weights"][i][1:]
            weights = [list(x) for x in zip(*weights)]
            for j in range(len(weights)):
                weights[j].append(case["initial_weights"][i][0][j])
            model.add(Layer(
                layer["number_of_neurons"],
                input_shape=case["model"].get('input_size'),
                activation=layer['activation_function'],
                weights=weights,
                bias=1,
            ))
        return model
    
    def load(self, path: str) -> Model:
        with open(path, 'rb') as f:
            return pickle.load(f)
