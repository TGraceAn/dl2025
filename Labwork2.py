import matplotlib.pyplot as plt


def linear_regression(x, y, F, lr, iter, threshold):
    assert len(x) == len(y), "length must be the same"
    W = [0, 0]
    loss_list = []
    weights_states = []

    loss = F(x, y, W)
    print(f"Initial loss: {loss}")
    loss_list.append(loss)
    print(f"Initial weights: {W}")

    for i in range(iter):
        # check if loss is less than threshold
        if loss < threshold:
            break
        # calculate gradient
        J_w0 = F_w0(x, y, W)
        J_w1 = F_w1(x, y, W)

        # update the weights
        W[0] = W[0] - lr * J_w0
        W[1] = W[1] - lr * J_w1

        # calculate loss
        loss = F(x, y, W)
        loss_list.append(loss)
        weights_states.append(W[:])
    
    return W, loss_list, weights_states

def F(x, y, W):
    J = 0
    for i in range(len(x)):
        J += (W[0] + W[1]*x[i] - y[i])**2
    J = J/(2*len(x))
    return J

def F_w0(x, y, W):
    J_w0 = 0
    N = len(x)
    for i in range(len(x)):
        J_w0 += (W[0] + W[1]*x[i] - y[i])
    J_w0 = J_w0/N
    return J_w0

def F_w1(x, y, W):
    J_w1 = 0 
    N = len(x)
    for i in range(len(x)):
        J_w1 += (W[0] + W[1]*x[i] - y[i])*x[i]
    J_w1 = J_w1/N
    return J_w1

def linear_regression_2(x, y, F, lr, iter, threshold):
    assert len(x) == len(y), "length must be the same"
    W = [0, 0]
    loss_list = []
    weights_states = []

    loss = F(x, y, W)
    print(f"Initial loss: {loss}")
    loss_list.append(loss)
    print(f"Initial weights: {W}")

    for i in range(iter):
        # check if loss is less than threshold
        if loss < threshold:
            break
        # calculate gradient
        W_0_h = [W[0] + 1e-5, W[1]]
        W_1_h = [W[0], W[1] + 1e-5]

        W_0_h_ = [W[0] - 1e-5, W[1]]
        W_0_h_ = [W[0], W[1] - 1e-5]

        J_w0 = (F(x, y, W_0_h) - F(x, y, W_0_h_)) / 2e-5
        J_w1 = (F(x, y, W_1_h) - F(x, y, W_0_h_)) / 2e-5

        # update the weights
        W[0] = W[0] - lr * J_w0
        W[1] = W[1] - lr * J_w1

        # calculate loss
        loss = F(x, y, W)
        loss_list.append(loss)
        weights_states.append(W[:])
        
    return W, loss_list, weights_states

	
def main():
    file = 'lr.csv'
    with open(file, 'r') as f:
        f = f.readlines()
        x = []
        y = []
        for data in f:
            values = data.strip().split(',')
            x.append(float(values[0]))
            y.append(float(values[1]))
        print(x)
        print(y)
    
    lr = 1e-4
    iter = 20

    W_linear_1, loss_list_1, states_1 = linear_regression(x, y, F, lr, iter, 1e-5)
    W_linear_2, loss_list_2, states_2 = linear_regression_2(x, y, F, lr, iter, 1e-5)
    # print(W_linear_1, loss_list_1[-1], states_1[-1])
    # print("")
    # print(W_linear_2, loss_list_2[-1], states_2[-1])

    plt.figure(figsize=(10, 5))
    plt.subplot(2, 2, 1)
    plt.plot(loss_list_1)
    plt.title("Linear Regression Loss (Analytic Gradient)")
    plt.xlabel(f"Iteration (Step), lr = {lr}")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.subplot(2, 2, 2)
    plt.plot(loss_list_2)
    plt.title("Linear Regression Loss (Numeric (Calulated) Gradient)")
    plt.xlabel(f"Iteration (Step), lr = {lr}")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.subplot(2, 2, 3)
    plt.plot(states_1)
    plt.title("Linear Regression Weights (Analytic Gradient)")
    plt.xlabel(f"Iteration (Step), lr = {lr}")
    plt.ylabel("Weights")
    plt.legend(["w0", "w1"])
    plt.grid(True)
    plt.subplot(2, 2, 4)
    plt.plot(states_2)
    plt.title("Linear Regression Weights (Numeric (Calulated) Gradient)")
    plt.xlabel(f"Iteration (Step), lr = {lr}")
    plt.ylabel("Weights")
    plt.legend(["w0", "w1"])
    plt.grid(True)
    plt.tight_layout()
    plt.show()


			    
if __name__ == "__main__":
    main()
