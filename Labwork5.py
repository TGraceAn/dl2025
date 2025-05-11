import matplotlib.pyplot as plt


class ModelConfig:
    n_layers = 2
    layers_size = [2, 2] # first is input, later are hidden
    n_class = 2

# Random #
class SimpleRandom:
    def __init__(self, seed=1):
        self.modulus = 2**32
        self.a = 1664525
        self.c = 1013904223
        self.state = seed

    def rand(self):
        self.state = (self.a * self.state + self.c) % self.modulus
        return self.state

    def rand_float(self):
        return self.rand() / self.modulus

    def uniform(self, low, high):
        return low + (high - low) * self.rand_float()

rng = SimpleRandom(seed=1162003)

# ----------- #
def e(terms=20):
    e = 0
    factorial = 1
    for n in range(terms):
        if n > 0:
            factorial *= n
        e += 1 / factorial
    return e

e_const = e()
# ----------- #
class ActFunction:
    """Interface for Activation Functions"""


class Sigmoid(ActFunction):
    """Sigmoid function"""
    def __init__(self):
        self.__value = None
        self.__grad = None

    def calValue(self, x):
        self.__value = 1/(1+e_const**(-x)) 
        return self.__value
    
    def calGrad(self):
        if self.__value == None:
            return "calValue must be called first"
        self.__grad = self.__value * (1 - self.__value)
        return self.__grad


# ----------- #
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


class BinaryCrossEntropy:
    def calLoss(self, target, pred):
        if isinstance(pred, list):
            pred = pred[0]
        if isinstance(target, list):
            target = target[0]
        pred = min(max(pred, 1e-15), 1 - 1e-15)
        J = target * log(pred) + (1 - target) * log(1 - pred)
        return -J

    def grad(self, target, pred):
        if isinstance(pred, list):
            pred = pred[0]
        if isinstance(target, list):
            target = target[0]
        pred = min(max(pred, 1e-15), 1 - 1e-15)
        return -(target / pred) + (1 - target) / (1 - pred) # formula, duh
    

# ----------- #
class Node:
    def __init__(self, n: int):
        """
        Arg:
            n (int): number of input feature
        """
        self.__bias = rng.uniform(0,1)
        self.__weights = [rng.uniform(0,1) for i in range(n)]
        self.__F = Sigmoid()

        # cache for efficient in backward
        self.__last_x = None
        self.__last_z = None
        self.__last_a = None

        self.__grad_weights = [0] * len(self.__weights)
        self.__grad_bias = 0

        # self.__weight_history = []
        # self.__bias_history = []

    def act(self, x):
        return self.__F.calValue(x)

    def forward(self, x):
        """
        Arg:
            x (list): Input features
        Return:
            Output of a single node
        """
        self.__last_x = x
        self.__last_z = sum(xi * wi for xi, wi in zip(x, self.__weights)) + self.__bias
        self.__last_a = self.act(self.__last_z)
        return self.__last_a

    def backward(self, target: int, loss_f):
        """
        Args: 
            target (int): target value (or dL_da)
            loss_f: loss function
        Return:
            Gradient for each of the weight
        """
        da_dz = self.__F.calGrad()
        dL_dz = target * da_dz
        grad_weights = [dL_dz * xi for xi in self.__last_x]
        grad_bias = dL_dz
        for i in range(len(self.__grad_weights)):
            self.__grad_weights[i] += grad_weights[i]
        self.__grad_bias += grad_bias
        grad_inputs = [dL_dz * w for w in self.__weights]
        return grad_weights, grad_bias, grad_inputs
    
    def step(self, lr):
        # self.__weight_history.append(self.__weights.copy())
        # self.__bias_history.append(self.__bias)

        self.__bias = self.__bias - lr * self.__grad_bias
        self.__weights = [(weight - lr * grad_weight) for weight, grad_weight in zip(self.__weights, self.__grad_weights)]

    def zero_grad(self):
        self.__grad_weights = [0] * len(self.__weights)
        self.__grad_bias = 0

    # @staticmethod
    # def show_weights_history(self):
    #     return self.__weight_history, self.__bias_history
    

class Layer:
    def __init__(self, n_in: int, n_out: int):
        """
        Arg:
            n_in (int): number of Inputs
            n_out (int): number of Outputs (number of nodes)
        """
        self.__nodes = [Node(n_in) for i in range(n_out)]

    def forward(self, x):
        """
        Arg:
            x (list): Output of the last layer or from feature
        Return:
            Output from all the nodes
        """
        out = []
        for node in self.__nodes:
            out.append(node.forward(x))

        return out

    def __len__(self):
        return len(self.__nodes)
    
    def backward(self, targets: list, loss_f):
        """
        Args: 
            target (list): list of target value (dL_das)
            loss_f: loss function
        Return:
            Every grads for each node
        """
        grad_inputs_all = []
        for node, dL_da in zip(self.__nodes, targets):
            grad_w, grad_b, grad_inputs = node.backward(dL_da, loss_f)
            grad_inputs_all.append(grad_inputs)
        grad_inputs_sum = [sum(grads[i] for grads in grad_inputs_all) for i in range(len(grad_inputs_all[0]))]
        return grad_inputs_sum
    
    def step(self, lr):
        for node in self.__nodes:
            node.step(lr)

    def zero_grad(self):
        for node in self.__nodes:
            node.zero_grad()

class FNN:
    def __init__(self, n: int, s: list, n_class = 2):
        """
        Arg:
            n (int): number of Hidden Layers
            s (list): the number of node(s) in each layer
            n_class (int): number of class to classify
        """
        self.__n_class = n_class

        if self.__n_class < 2:
            print("This is dumb")
            print("Nothing initialized")
        else:
            print(f"Init a model with {n} Hidden Layers")
            self.__layers = [Layer(s[i], s[i+1]) for i in range(n-1)]
            self.__out = Layer(s[-1], 1 if n_class == 2 else n_class)
            # tbh the last layer should be softmax smth for multiple classes but this here for convention for now

        self.__last_x = None

    def forward(self, x):
        """
        Return: 
            Return the output
        """
        for layer in self.__layers:
            x = layer.forward(x)

        if self.__n_class == 2:
            out = self.__out.forward(x)
            return out
        else:
            x = self.__out.forward(x)
            # nomalization
            total = sum(x)
            out = [value/total for value in x]
            return out

    def __len__(self):
        return len(self.__layers) + 1 # return value of layers
    
    def backward(self, target, loss_f):
        out = self.forward(self.__last_x)
        if self.__n_class == 2:
            dL_da = loss_f.grad(target, out)
            grad_inputs = self.__out.backward([dL_da], loss_f)
        else:
            dL_das = [loss_f.grad(t, p) for t, p in zip(target, out)]
            grad_inputs = self.__out.backward(dL_das, loss_f)
        
        for layer in reversed(self.__layers):
            grad_inputs = layer.backward(grad_inputs, loss_f)

    def step(self, lr):
        for layer in self.__layers:
            layer.step(lr)
        self.__out.step(lr)

    def zero_grad(self):
        for layer in self.__layers:
            layer.zero_grad()
        self.__out.zero_grad()

if __name__ == "__main__":
    config = ModelConfig()
    fnn = FNN(config.n_layers, config.layers_size, config.n_class)
    print(f"Number of layers: {len(fnn)}")

    # XOR problem
    x = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y = [0, 1, 1, 0]  # target

    # Training
    loss_f = BinaryCrossEntropy()
    epochs = 1000
    lr = 0.1
    loss_history = []

    for epoch in range(epochs):
        total_loss = 0
        fnn.zero_grad()
        for xi, yi in zip(x, y):
            out = fnn.forward(xi)
            loss = loss_f.calLoss(yi, out)
            total_loss += loss
            fnn.backward(yi, loss_f)
        fnn.step(lr)
        avg_loss = total_loss / len(x)
        loss_history.append(avg_loss)
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {avg_loss:.6f}")

    print("Training complete.")

    # Plot loss history
    plt.figure(figsize=(8, 6))
    plt.plot(loss_history)
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('loss_history.png')


