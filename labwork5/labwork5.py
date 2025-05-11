import random
import csv
import math

class Layer:
    def __init__(self, num_neurons, num_inputs):
        self.weights = [[random.uniform(-1, 1) for _ in range(num_inputs)] for _ in range(num_neurons)]
        self.biases = [[random.uniform(-1, 1)] for _ in range(num_neurons)]
        self.outputs = []
        self.d_weights = []
        self.d_biases = []

    def feedforward(self, inputs):
        self.inputs = inputs
        self.outputs = [
            [self.activate(sum(w * i for w, i in zip(weight, inputs)) + bias[0])]
            for weight, bias in zip(self.weights, self.biases)
        ]
        return self.outputs

    def activate(self, x):
        return 1 / (1 + math.exp(-x))

    def activate_derivative(self, x):
        return x * (1 - x)


class NeuralNetwork:
    def __init__(self, layers_config, learning_rate=0.01):
        self.layers = []
        self.learning_rate = learning_rate

        for i in range(1, len(layers_config)):
            self.layers.append(Layer(layers_config[i], layers_config[i - 1]))

    def feedforward(self, inputs):
        for layer in self.layers:
            inputs = [x[0] for x in layer.feedforward(inputs)]
        return inputs

    def backpropagate(self, targets):
        error = [(target - output[0]) for target, output in zip(targets, self.layers[-1].outputs)]
        layer_deltas = []

        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            if i == len(self.layers) - 1:
                delta = [e * layer.activate_derivative(o[0]) for e, o in zip(error, layer.outputs)]
            else:
                next_layer = self.layers[i + 1]
                delta = [
                    sum(
                        next_delta * next_weight
                        for next_delta, next_weight in zip(layer_deltas[0], [weights[j] for weights in next_layer.weights])
                    )
                    * layer.activate_derivative(o[0])
                    for j, o in enumerate(layer.outputs)
                ]

            layer_deltas.insert(0, delta)
            # Update weights and biases
            for j, neuron_delta in enumerate(delta):
                for k, input_val in enumerate(layer.inputs):
                    layer.weights[j][k] += self.learning_rate * neuron_delta * input_val
                layer.biases[j][0] += self.learning_rate * neuron_delta

    def train(self, X, y, epochs):
        for epoch in range(epochs):
            for inputs, targets in zip(X, y):
                self.feedforward(inputs)
                self.backpropagate(targets)
            if epoch % 100 == 0 or epoch == epochs - 1:
                loss = sum(
                    (target - output) ** 2 for output, target in zip(self.feedforward(inputs), targets)
                ) / len(targets)
                print(f"Epoch {epoch}: Loss = {loss:.4f}")


def read_csv(file_path):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        data = list(reader)

    X = [[float(x) for x in row[:-1]] for row in data]
    y = [[float(row[-1])] for row in data]

    return X, y


if __name__ == "__main__":
    file_path = "labwork4/file.txt"
    X, y = read_csv(file_path)

    layers_config = [len(X[0]), 5, 1]
    learning_rate = 0.01
    epochs = 1000

    neural_net = NeuralNetwork(layers_config, learning_rate=learning_rate)
    neural_net.train(X, y, epochs)

    for inputs in X:
        output = neural_net.feedforward(inputs)
        print(f"Input: {inputs} -> Predicted Output: {output}")
