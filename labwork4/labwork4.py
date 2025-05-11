import random

class Layer:
    def __init__(self, num_neurons, num_inputs, initialize_random=True, weights=None, biases=None):
        if initialize_random:
            self.weights = [[random.random() for _ in range(num_inputs)] for _ in range(num_neurons)]
            self.biases = [[random.random()] for _ in range(num_neurons)]
        else:
            self.weights = weights if weights is not None else [[0 for _ in range(num_inputs)] for _ in range(num_neurons)]
            self.biases = biases if biases is not None else [[0] for _ in range(num_neurons)]

        print(f"Initialized Layer with {num_neurons} neurons, each with {num_inputs} inputs")
        print("Weights:", self.weights)
        print("Biases:", self.biases)

    def feedforward(self, inputs):
        outputs = []
        for weight, bias in zip(self.weights, self.biases):
            weighted_sum = sum(w * i for w, i in zip(weight, inputs)) + bias[0]
            outputs.append([weighted_sum])
        return outputs

class NeuralNetwork:
    def __init__(self, layers_config, initialize_random=True, weights_bias_file=None):
        self.layers = []

        for i in range(1, len(layers_config)):
            num_inputs = layers_config[i - 1]
            num_neurons = layers_config[i]

            if weights_bias_file:
                with open(weights_bias_file, 'r') as file:
                    weights = []
                    biases = []
                    for _ in range(num_neurons):
                        weights.append(list(map(float, file.readline().strip().split())))
                        biases.append([float(file.readline().strip())])

                self.layers.append(Layer(num_neurons, num_inputs, initialize_random=False, weights=weights, biases=biases))
            else:
                self.layers.append(Layer(num_neurons, num_inputs, initialize_random=initialize_random))

    def feedforward(self, inputs):
        activations = [[i] for i in inputs]
        for layer in self.layers:
            activations = layer.feedforward([a[0] for a in activations])
        return activations

def read_network_config(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    num_layers = int(lines[0].strip())
    layers_config = [int(lines[i].strip()) for i in range(1, num_layers + 1)]

    return layers_config

if __name__ == "__main__":
    config_file = "labwork4/file.txt"
    weights_file = None

    layers_config = read_network_config(config_file)
    print("Network Configuration:", layers_config)

    neural_net = NeuralNetwork(layers_config, initialize_random=True, weights_bias_file=weights_file)

    test_input = [1, 2, 3]
    output = neural_net.feedforward(test_input)

    print("Input:", test_input)
    print("Output:", output)
