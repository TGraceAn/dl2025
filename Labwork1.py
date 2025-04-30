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
        L=3,
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
    print(result)
    print(fx)
    print(x_s)

    plt.figure(figsize=(10, 5))
    plt.subplot(3, 1, 1)
    plt.plot(x_s)
    plt.title("Gradient Descent (x Value)")
    plt.xlabel("Iteration (Step)")
    plt.ylabel("x Value")
    plt.grid(True)
    plt.subplot(3, 1, 2)
    plt.plot(fx)
    plt.title("Gradient Descent (Gradient)")
    plt.xlabel("Iteration (Step)")
    plt.ylabel("f(x)")
    plt.grid(True)

    # Plot the function f(x) = x^2 with values of x dotted using x_s with visualization of how x_s changes as lines 
    x = [i for i in range(-10, 11)]
    y = [f_x_squared(i) for i in x]
    plt.subplot(3, 1, 3)
    plt.plot(x, y)
    plt.scatter(x_s, [f_x_squared(i) for i in x_s], color='red', label='x_s values')
    for i in range(len(x_s)-1):
        plt.plot([x_s[i], x_s[i+1]], [f_x_squared(x_s[i]), f_x_squared(x_s[i+1])], linestyle='--', color='gray')

    plt.title("Gradient Descent (Function Plot)")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.savefig("gradient_descent_lr(0).png")

