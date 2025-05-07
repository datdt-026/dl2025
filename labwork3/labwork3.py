import numpy as np
import pandas as pd


def load_data(file_path):
    data = pd.read_csv(file_path)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    return X, y


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def compute_loss(y, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

def logistic_regression(X, y, lr=0.01, epochs=1000):

    X = np.hstack((np.ones((X.shape[0], 1)), X))


    w = np.zeros(X.shape[1])

    for epoch in range(epochs):
        z = np.dot(X, w)
        y_pred = sigmoid(z)

        gradient = np.dot(X.T, (y_pred - y)) / y.size

        w -= lr * gradient

        if epoch % 100 == 0 or epoch == epochs - 1:
            loss = compute_loss(y, y_pred)
            print(f"Epoch {epoch}: Loss = {loss}")

    return w

# Main
if __name__ == "__main__":
    file_path = "/Users/dat/Desktop/USTH_Master/deep_learning_25/dl2025/labwork3/loan.csv"

    X, y = load_data(file_path)

    learning_rate = 0.1
    epochs = 1000
    weights = logistic_regression(X, y, lr=learning_rate, epochs=epochs)

    print(f"Final weights: w0 = {weights[0]}, w1 = {weights[1]}, w2 = {weights[2]}")
