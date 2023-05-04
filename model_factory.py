from json_reader import JsonReader
from ann import Model, Layer

class ModelFactory:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.json_reader = JsonReader(f'{model_name}')
        self.model = None

    def create(self) -> Model:
        self.json_reader.read()
        case = self.json_reader.get("case")
        print(case)
        learning_parameters = case["learning_parameters"]
        self.model = Model(
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
            self.model.add(Layer(
                layer["number_of_neurons"],
                input_shape=case["model"].get('input_size'),
                activation=layer['activation_function'],
                weights=weights,
                bias=1,
            ))
        return self.model