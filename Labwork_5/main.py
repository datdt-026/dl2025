import random
import math

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# Derivative of the sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)

# Initialize weights
def initialize_weights(input_size, hidden_size, output_size):
    hidden_weights = [[random.uniform(-1, 1) for _ in range(hidden_size)] for _ in range(input_size)]
    output_weights = [random.uniform(-1, 1) for _ in range(hidden_size)]
    return hidden_weights, output_weights

# Initialize biases
def initialize_biases(hidden_size, output_size):
    hidden_bias = [random.uniform(-1, 1) for _ in range(hidden_size)]
    output_bias = random.uniform(-1, 1)
    return hidden_bias, output_bias

# Forward pass
def forward_pass(X, hidden_weights, output_weights, hidden_bias, output_bias):
    hidden_layer_input = [sum(X[i] * hidden_weights[i][j] for i in range(len(X))) + hidden_bias[j] for j in range(len(hidden_bias))]
    hidden_layer_output = [sigmoid(x) for x in hidden_layer_input]

    output_layer_input = sum(hidden_layer_output[j] * output_weights[j] for j in range(len(hidden_layer_output))) + output_bias
    output_layer_output = sigmoid(output_layer_input)

    return hidden_layer_output, output_layer_output

# Backpropagation
def backpropagation(X, y, hidden_layer_output, output_layer_output, hidden_weights, output_weights, hidden_bias, output_bias, learning_rate):
    output_error = y - output_layer_output
    output_delta = output_error * sigmoid_derivative(output_layer_output)

    hidden_error = [output_delta * output_weights[j] * sigmoid_derivative(hidden_layer_output[j]) for j in range(len(hidden_layer_output))]
    hidden_delta = [error * sigmoid_derivative(hidden_layer_output[j]) for j, error in enumerate(hidden_error)]

    for i in range(len(hidden_weights)):
        for j in range(len(hidden_weights[0])):
            hidden_weights[i][j] += learning_rate * X[i] * hidden_delta[j]

    for j in range(len(output_weights)):
        output_weights[j] += learning_rate * hidden_layer_output[j] * output_delta

    for j in range(len(hidden_bias)):
        hidden_bias[j] += learning_rate * hidden_delta[j]

    output_bias += learning_rate * output_delta

# Calculate loss
def calculate_loss(y_true, y_pred):
    return 0.5 * (y_true - y_pred) ** 2

# Train the neural network
def train(X, y, hidden_size, output_size, epochs, learning_rate, print_interval):
    input_size = len(X[0])
    hidden_weights, output_weights = initialize_weights(input_size, hidden_size, output_size)
    hidden_bias, output_bias = initialize_biases(hidden_size, output_size)

    for epoch in range(epochs):
        total_loss = 0
        for i in range(len(X)):
            hidden_layer_output, output_layer_output = forward_pass(X[i], hidden_weights, output_weights, hidden_bias, output_bias)
            loss = calculate_loss(y[i], output_layer_output)
            total_loss += loss
            backpropagation(X[i], y[i], hidden_layer_output, output_layer_output, hidden_weights, output_weights, hidden_bias, output_bias, learning_rate)

        avg_loss = total_loss / len(X)
        if epoch % print_interval == 0:
            print("Epoch:", epoch, "Average Loss:", avg_loss)

    return hidden_weights, output_weights, hidden_bias, output_bias

# Test the trained neural network
def test(X, hidden_weights, output_weights, hidden_bias, output_bias):
    predictions = []
    for i in range(len(X)):
        _, output_layer_output = forward_pass(X[i], hidden_weights, output_weights, hidden_bias, output_bias)
        print("Input:", X[i], "Predicted Output:", output_layer_output)
        predictions.append(output_layer_output)
    return predictions

# Example usage
X = [[0, 0], [0, 1], [1, 0], [1, 1]]
y = [0, 1, 1, 0]
hidden_weights, output_weights, hidden_bias, output_bias = train(X, y, hidden_size=2, output_size=1, epochs=10000, learning_rate=0.1, print_interval=1000)
predictions = test(X, hidden_weights, output_weights, hidden_bias, output_bias)
print("Final Predictions:", predictions)
