import matplotlib.pyplot as plt


class ModelConfig:
    n_layers = 4
    layers_size = [2, 3, 3, 5] # first is input, later are hidden
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
def e(terms=5):
    e = 0
    factorial = 1
    for n in range(terms):
        if n > 0:
            factorial *= n
        e += 1 / factorial
    return e

# ----------- #
class Node:
    def __init__(self, n: int):
        """
        Arg:
            n (int): number of input feature
        """
        self.bias = rng.uniform(0,1)
        self.weights = [rng.uniform(0,1) for i in range(n)]

    def forward(self, x):
        """
        Arg:
            x (list): Input features
        Return:
            Output of a single node
        """
        act = self.act
        total = 0
        for i in range(len(x)):
            total += x[i] * self.weights[i]
        total += self.bias
        out = act(total)
        return out

    def act(self, x):
        return 1/(1+e()**(-x)) 
    

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
        elif self.__n_class == 2:
            print(f"Init a model with {n} Hidden Layers")
            self.__layers = [Layer(s[i], s[i+1]) for i in range(n-1)]
            self.__out = Layer(s[-1], 1)
        else:
            print(f"Init a model with {n} Hidden Layers")
            self.__layers = [Layer(s[i], s[i+1]) for i in range(n-1)]
            # tbh the last layer should be softmax smth but this here for convention for now
            self.__out = Layer(s[-1], n_class)

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


if __name__ == "__main__":
    config = ModelConfig()
    fnn = FNN(config.n_layers, config.layers_size, config.n_class)
    print(len(fnn))
    x = [1, 1]
    print(fnn.forward(x))