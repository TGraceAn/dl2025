import matplotlib.pyplot as plt

def logistic_regression_3D(x: list, y: list, F, lr: float, iter: int, W: list, threshold: float):
    """
    Args:
        x (list) : A list of features (Here, 2 features)
        y (list) : True or False (0, 1)
        F : Loss function (cost function)
        lr (float): Learning rate
        iter (int): Max iteration
        W (list): list of initial weights
        threshold (float): The value in which if the update is too small, break.
    Return: 
        List of the trained weights and some other stuff.
    """
    assert len(x) == len(y), "Make sure the data and label length are the same size"

    loss_list = []
    weights_states = []

    loss = F(x, y, W)
    print(f"Initial loss: {loss}")
    loss_list.append(loss)
    print(f"Initial weights: {W}")

    for i in range(iter):
        # calculate the gradient
        J_w0 = BCE_w0(x, y, W)
        J_w1 = BCE_w1(x, y, W)
        J_w2 = BCE_w2(x, y, W)

        # update the weights
        W[0] = W[0] - lr * J_w0
        W[1] = W[1] - lr * J_w1
        W[2] = W[2] - lr * J_w2
        
        # break if the update is too small
        loss = F(x, y, W)
        loss_list.append(loss)
        if abs(loss_list[-1] - loss_list[-2]) < threshold:
            break

    return W, loss_list, weights_states

def y_hat(x, W):
    return W[0] + W[1]*x[0] + W[2]*x[1]

def BCE(x, y, W):
    J = 0
    for i in range(len(x)):
        J += y[i]*y_hat(x[i], W) - log(1 + e()**(y_hat(x[i], W)))
    J = -J/len(x)
    return J

def BCE_w0(x, y, W):
    J_w0 = 0
    N = len(x)
    for i in range(N):
        J_w0 += 1 - y[i] - sigmoid(-y_hat(x[i], W))
    J_w0 = J_w0/N
    return J_w0

def BCE_w1(x, y, W):
    J_w1 = 0 
    N = len(x)
    for i in range(len(x)):
        J_w1 += - y[i]*x[i][0] + x[i][0]*(1 - sigmoid(-y_hat(x[i], W))) # 0 is 1
    J_w1 = J_w1/N
    return J_w1

def BCE_w2(x, y, W):
    J_w2 = 0 
    N = len(x)
    for i in range(len(x)):
        J_w2 += -y[i]*x[i][1] + x[i][1]*(1 - sigmoid(-y_hat(x[i], W))) # 0 is 1
    J_w2 = J_w2/N
    return J_w2

def e(terms=5):
    e = 0
    factorial = 1
    for n in range(terms):
        if n > 0:
            factorial *= n
        e += 1 / factorial
    return e

def log(x, terms=10):
    if x <= 0:
        print("log(x) is undefined for x <= 0")

    # https://en.wikipedia.org/wiki/Mercator_series + tanh

    # Bring x to the range [1, 2) -> fast converge
    n = 0
    while x > 2:
        x /= 2
        n += 1
    while x < 1:
        x *= 2
        n -= 1

    # log(x) = 2 * atanh((x - 1)/(x + 1))
    # https://en.wikipedia.org/wiki/Mercator_series
    y = (x - 1) / (x + 1)

    result = 0
    for k in range(terms):
        term = (1 / (2 * k + 1)) * (y ** (2 * k + 1))
        result += term

    ln_x = 2 * result
    ln2 = 0.6931471805599453  # ln(2) hardcoded

    return ln_x + n * ln2

def sigmoid(x):
    return 1/(1+e()**(-x))

def load_csv(file, header=True):
    with open(file, 'r') as f:
        f = f.readlines()
        if header == True:
            f = f[1:]
        f1 = []
        f2 = []
        y = []
        for data in f:
            values = data.strip().split(',')
            f1.append(float(values[0]))
            f2.append(float(values[1]))
            y.append(float(values[2]))

        x = []
        for data in zip(f1, f2):
            x.append(data)
        # print(x)
        # print(y)
    return x, y

def main():
    file = "loan2.csv"
    x, y = load_csv(file)
    W = [0, 0, 0]
    lr = 3e-4
    iter = 20

    W_logistic_1, loss_list_1, states_1 = logistic_regression_3D(x, y, BCE, lr, iter, W, 1e-5)
    print(W_logistic_1)

if __name__ == "__main__":
    main()
