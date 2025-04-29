
def f(x):
    return x**2

def f_prime(x):
    return 2 * x

def gradient_descent(starting_point, num_iterations, learning_rate=None):
    if learning_rate is None:
        learning_rate = 0.1

    current_point = starting_point
    print("Initial formula f(x): x^2")

    print("Iteration\t\tx\t\tf(x)")
    for i in range(num_iterations):
        gradient = f_prime(current_point)
        current_point -= learning_rate * gradient
        fx = f(current_point)
        print(f"Iteration {i+1}:\t {current_point:.6f}\t {fx:.6f}")

    return current_point, f(current_point)

starting_point = 10
num_iterations = 10
learning_rate = float(input("Enter learning rate (default is 0.1 if not provided): ") or 0.1)

min_point, min_value = gradient_descent(starting_point, num_iterations, learning_rate)
print("\nMinimum point:", min_point)
print("Minimum value:", min_value)
