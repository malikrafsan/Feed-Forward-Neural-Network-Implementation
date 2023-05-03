import numpy as np
from typing import Callable
import activations
import networkx as nx
import matplotlib.pyplot as plt
import random


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
        return (self.input_shape + 1) * len(self.neurons) 

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
        param_count = self.get_params_count()

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

    def draw(self):
        graph = nx.DiGraph()

        # draw input layer
        num_inputs = self.layers[0].input_shape
        for i in range(num_inputs):
            graph.add_node(f'i_{i}', color='green', pos=(0,-i))

        # draw hidden layers
        for i, layer in enumerate(self.layers):
            graph.add_node(f'b_{i}', color='black', pos=(i + 0.25, -layer.input_shape))
            for j, neuron in enumerate(layer.neurons):
                color = 'blue' if i < len(self.layers) - 1 else 'red'
                cur_node = f'h_{i}_{j}' if i < len(self.layers) - 1 else f'o_{j}'
                graph.add_node(cur_node, color=color, pos=(i+1, -j + random.randrange(-3, 3) * 0.1))

                for k, weight in enumerate(neuron.weights):
                    src = f'b_{i}' if k == len(neuron.weights) - 1 else (f'i_{k}' if i == 0 else f'h_{i-1}_{k}')
                    graph.add_edge(src, cur_node, weight=weight)
            
        # draw graph
        pos = nx.get_node_attributes(graph, 'pos')
        edge_labels = nx.get_edge_attributes(graph, 'weight')

        # make plot bigger
        plot = plt.gca()
        plot.figure.set_size_inches(10, 10)
        plot.set_title('Feed Forward Neural Network Model')

        nx.draw(graph, pos, with_labels=True, node_color=[graph.nodes[node].get('color') for node in graph.nodes], node_size=1000, font_size=8, font_color='white', ax=plot)
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=8, ax=plot)


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
