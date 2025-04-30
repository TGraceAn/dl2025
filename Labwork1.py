import matplotlib.pyplot as plt

def gradient_descent(f, x_0, h, L, max_iterations, threshold = 1e-6):
    fx = []
    x_s = []
    for i in range(max_iterations):
        derivative = (f(x_0 + h) - f(x_0 - h)) / (2*h)
        x_a = x_0 - L * derivative
        f_xa = f(x_a)
        fx.append(f_xa)
        x_s.append(x_a)
        print(f"Step {i + 1}: x = {x_a:.6f} | f(x) = {f(x_a):.2f}")

        if abs(f(x_a) - f(x_0)) < threshold:
            break
        x_0 = x_a

    return x_0, fx, x_s

def f_x_squared(x: float) -> float:
    return x ** 2

if __name__ == "__main__":
    result, fx, x_s = gradient_descent(
        f=f_x_squared,
        x_0=10.0,
        h=0.001,
        L=0.1,
        max_iterations=10
    )

    # plt.plot(x_s)
    # plt.title("Gradient Descent (x Value)")
    # plt.xlabel("Iteration (Step)")
    # plt.ylabel("x Value")
    # plt.grid(True)
    # plt.show()

    # plt.plot(fx)
    # plt.title("Gradient Descent (Gradient)")
    # plt.xlabel("Iteration (Step)")
    # plt.ylabel("f(x)")
    # plt.grid(True)
    # plt.show()

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x_s)
    plt.title("Gradient Descent (x Value)")
    plt.xlabel("Iteration (Step)")
    plt.ylabel("x Value")
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(fx)
    plt.title("Gradient Descent (Gradient)")
    plt.xlabel("Iteration (Step)")
    plt.ylabel("f(x)")
    plt.grid(True)
    plt.show()

