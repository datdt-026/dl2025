import pandas as pd

def load_data(file_path):
    data = pd.read_csv(file_path)
    x = data['x'].tolist()
    y = data['y'].tolist()
    x_max = max(x)
    x_min = min(x)
    x = [(xi - x_min) / (x_max - x_min) for xi in x]

    return x, y

def compute_cost(x, y, w0, w1):
    m = len(x)
    total_error = 0
    for i in range(m):
        prediction = w0 + w1 * x[i]
        total_error += (prediction - y[i]) ** 2
    cost = total_error / (2 * m)
    return cost

#
def gradient_descent(x, y, w0, w1, learning_rate, iterations):
    m = len(x)
    for i in range(iterations):
        total_error_w0 = 0
        total_error_w1 = 0
        for j in range(m):
            prediction = w0 + w1 * x[j]
            total_error_w0 += prediction - y[j]
            total_error_w1 += (prediction - y[j]) * x[j]


        w0 -= learning_rate * (total_error_w0 / m)
        w1 -= learning_rate * (total_error_w1 / m)


        if i % 100 == 0:
            cost = compute_cost(x, y, w0, w1)
            print(f"Iteration {i}, Cost: {cost}, w0: {w0}, w1: {w1}")

    return w0, w1

def main(file_path, learning_rate=0.0001, iterations=800):
    x, y = load_data(file_path)
    w0 = 2
    w1 = 0

    w0, w1 = gradient_descent(x, y, w0, w1, learning_rate, iterations)

    print(f"Final values: w0 = {w0}, w1 = {w1}")

file_path = 'labwork2/lr.csv'
main(file_path)
