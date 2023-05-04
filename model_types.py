from typing import TypedDict

class Layers(TypedDict):
    number_of_neurons: int
    activation_function : str

class ModelData(TypedDict):
    input_size: int
    layers: Layers

class LearningParameters(TypedDict): 
    learning_rate: float
    batch_size: int
    max_iteration: int
    error_threshold: float

class Case(TypedDict):
    model: ModelData
    input: list[list[list[float]]]
    initial_weights: list[list[list[float]]]
    target: list[list[list[float]]]
    learning_parameters: LearningParameters

class Expect(TypedDict):
    stopped_by: str


class ModelConfig(TypedDict):
    case: Case
    expect: Expect
