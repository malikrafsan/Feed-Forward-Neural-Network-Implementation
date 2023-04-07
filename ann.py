import numpy as np
from typing import Callable
import activations
import networkx as nx
import matplotlib.pyplot as plt


class Neuron(object):
    def __init__(
        self,
        activation: Callable | str = 'sigmoid',
        weights: list[float] = None,
    ):
        self.activation = activation
        if isinstance(activation, str):
            self.activation: Callable = getattr(activations, activation)
        self.weights = weights
        self.value = 0

    def __call__(self, x: list[float]):
        # feed forward
        val = np.dot(x, self.weights)
        self.value = self.activation(val)
        return self.value

    def __repr__(self):
        return f'Neuron({self.activation.__name__}, {self.weights})'


class Layer(object):
    def __init__(
        self,
        neurons: list[Neuron] | int,
        name: str = '',
        activation: Callable | str = '',
        weights: list[list[float]] = None,
        bias: float = 1,
        input_shape=0,
    ) -> None:
        self.name = name
        self.activation = activation
        if isinstance(activation, str):
            self.activation: Callable = getattr(activations, activation)
        self.neurons = neurons
        if isinstance(neurons, int):
            self.neurons: list[Neuron] = [
                Neuron(activation, weights[i] if weights else 0)
                for i in range(neurons)
            ]
        self.bias = bias
        self.input_shape = input_shape

    def get_output_shape(self):
        return len(self.neurons)

    def get_params_count(self):
        return (len(self.neurons) + 1) * self.input_shape

    def get_weights(self):
        return np.array([neuron.weights for neuron in self.neurons])

    def __call__(self, inputs: list[float]) -> list[float]:
        # feed forward
        out = [neuron(inputs) for neuron in self.neurons]
        if self.activation is activations.softmax:
            out = (out / sum(out)).tolist()
        return out

    def __repr__(self):
        activation_name = self.activation.__name__
        weights = self.get_weights()
        param_count = (len(weights) + 1) * self.input_shape

        return ''.join([
            'Layer(',
            f'activation={activation_name},',
            f'weights={weights},',
            f'bias={self.bias},',
            f'param_count={param_count}',
            ')',
        ])


class Model(object):
    def __init__(self, layers: list[Layer] = None) -> None:
        self.layers = layers
        if layers is None:
            self.layers: list[Layer] = []

    def add(self, layer: Layer) -> None:
        if self.layers:
            layer.input_shape = self.layers[-1].get_output_shape()
        self.layers.append(layer)

    def get_params_count(self):
        return sum([layer.get_params_count() for layer in self.layers])

    def summary(self):
        print(f'Model with {self.get_params_count()} parameters')
        for layer in self.layers:
            print(layer)

    def __call__(self, inputs: list[list[float]]):
        # feed forward
        outputs: list[list[float]] = []
        for input in inputs:
            out = input
            for layer in self.layers:
                out.append(1)
                out = layer(out)
            outputs.append(out)
        
        return outputs
    
    def draw(self):
        graph = nx.DiGraph()

        # draw input layer
        num_neurons = len(self.layers[0].neurons[0].weights) -1
        print("num_neurons", num_neurons)

        for i in range(num_neurons):
            graph.add_node(f"-1-{i}", pos=(-1, i))

        for i, layer in enumerate(self.layers):
            for j, neuron in enumerate(layer.neurons):
                count = 0

                graph.add_node(f"{i}-{j}", pos=(i, j))
                print("neuron.weights",)
                for k, weight in enumerate(neuron.weights):
                    graph.add_edge(f"{i-1}-{k}", f"{i}-{j}", weight=weight)
                    count += 1
                graph.add_edge(f"{i-1}-{count}", f"{i}-{j}", weight=layer.bias)

        pos = nx.get_node_attributes(graph, 'pos')
        nx.draw(graph, pos, with_labels=True)
        edge_labels = nx.get_edge_attributes(graph, 'weight')
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)
        plt.show()


if __name__ == '__main__':
    model = Model()
    model.add(Layer(
        3,
        input_shape=4,
        activation='relu',
        weights=[0.5, 0.5, 1],
        bias=1,
    ))
    model.add(Layer(
        2,
        activation='sigmoid',
        weights=[0.5, 0.5],
        bias=1,
    ))
    model.summary()
