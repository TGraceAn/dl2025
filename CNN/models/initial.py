from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Union
import math


"""Initial stuff for CNN"""

# Module (model, layer, etc) #
class Module(ABC):
    """Interface for all modules"""

    @abstractmethod
    def __init__(self):
        """Initialize the module"""

    @abstractmethod
    def forward(self, x):
        """Forward ..."""
        
    @abstractmethod
    def backward(self, x):
        """Backward ..."""

    @abstractmethod
    def step(self, lr):
        """Update the weights"""

    @abstractmethod
    def zero_grad(self):
        """Zero the gradients"""


class Convol(Module):
    """Convolutional layer"""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def forward(self, x):
        pass

    def backward(self, x):
        pass

    def step(self, lr):
        pass

    def zero_grad(self):
        pass


class MaxPooling(Module):
    """Max pooling layer"""
    def __init__(self, kernel_size: int, stride: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        pass

    def backward(self, x):
        pass

    def step(self, lr):
        pass

    def zero_grad(self):
        pass


class Dense(Module):
    """Dense layer"""
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x):
        pass

    def backward(self, x):
        pass

    def step(self, lr):
        pass

    def zero_grad(self):
        pass


# Optimizer stuff #
class Optimizer(ABC):
    """Optimizer for the module"""

    @abstractmethod
    def __init__(self, module: Module, lr: float):
        """Initialize the optimizer"""
        self.module = module
        self.lr = lr

    @abstractmethod
    def step(self):
        """Update the weights"""
        self.module.step(self.lr)

    @abstractmethod
    def zero_grad(self):
        """Zero the gradients"""
        self.module.zero_grad()
        

# Activation Functions #
class ActFunction(ABC):
    """Interface for Activation Functions"""
    @abstractmethod
    def __init__(self):
        """Initialize the activation function"""

    @abstractmethod
    def calValue(self, x):
        """Calculate the value of the activation function"""
        
    @abstractmethod
    def calGrad(self):
        """Calculate the gradient of the activation function"""


class Sigmoid(ActFunction):
    """Sigmoid function"""
    def __init__(self):
        self.__value = None
        self.__grad = None

    def calValue(self, x):
        self.__value = 1/(1+math.exp**(-x)) 
        return self.__value
    
    def calGrad(self):
        if self.__value is None:
            return "calValue must be called first"
        self.__grad = self.__value * (1 - self.__value)
        return self.__grad
    

class Relu(ActFunction):
    """ReLU function"""
    def __init__(self):
        self.__value = None
        self.__grad = None

    def calValue(self, x):
        self.__value = max(0, x)
        return self.__value
    
    def calGrad(self):
        if self.__value is None:
            return "calValue must be called first"
        if self.__value > 0:
            self.__grad = 1
        else:
            self.__grad = 0
        return self.__grad
    
# Loss #
class Loss(ABC):
    """Interface for Loss Functions"""
    @abstractmethod
    def calLoss(self, target, pred):
        """Calculate the loss"""

    @abstractmethod
    def grad(self, target, pred):
        """Calculate the gradient of the loss"""


# TODO: Replace later for more beautiful code
class BinaryCrossEntropy(Loss):
    def calLoss(self, target, pred):
        if isinstance(pred, list):
            pred = pred[0] # taking the value only
        if isinstance(target, list):
            target = target[0] # taking the value only
        pred = min(max(pred, 1e-15), 1 - 1e-15)
        J = target * math.log(pred) + (1 - target) * math.log(1 - pred)
        return -J

    # Use for gradient descent
    def grad(self, target, pred):
        if isinstance(pred, list):
            pred = pred[0] # taking the value only
        if isinstance(target, list):
            target = target[0] # taking the value only
        pred = min(max(pred, 1e-15), 1 - 1e-15)
        return -(target / pred) + (1 - target) / (1 - pred) # formula, duh
    
# Previous Node class from labwork5 now's gonna be Tensor #
class Tensor:
    """Tensor class"""
    def __init__(self, data, requires_grad=True):
        self.data = data if isinstance(data, list) else [data]
        self.requires_grad = requires_grad
        self.grad = None

    def __getitem__(self, index):
        value = self.data[index]
        return Tensor(value) if not isinstance(value, list) else Tensor(value)

    ## Operators for tensor operations ## 
    def __add__(self, other):
        if isinstance(other, Tensor):
            return Tensor([x + y for x, y in zip(self.data, other.data)])
        else:
            return Tensor([x + other for x in self.data])
        
    def __sub__(self, other):
        if isinstance(other, Tensor):
            return Tensor([x - y for x, y in zip(self.data, other.data)])
        else:
            return Tensor([x - other for x in self.data])
        
    def __mul__(self, other):
        if isinstance(other, Tensor):
            return Tensor([x * y for x, y in zip(self.data, other.data)])
        else:
            return Tensor([x * other for x in self.data])
        
    def __truediv__(self, other):
        if isinstance(other, Tensor):
            return Tensor([x / y for x, y in zip(self.data, other.data)])
        else:
            return Tensor([x / other for x in self.data])
        
    def __pow__(self, other):
        if isinstance(other, Tensor):
            return Tensor([x ** y for x, y in zip(self.data, other.data)])
        else:
            return Tensor([x ** other for x in self.data])
        
    def __neg__(self):
        if not isinstance(self.data, list):
            return Tensor(-self.data)
        return Tensor([-x for x in self.data])
    
    def __radd__(self, other):
        if isinstance(other, Tensor):
            return self + other
        else:
            return self + other
        
    def __rmul__(self, other):
        if isinstance(other, Tensor):
            return self * other
        else:
            return self * other
        
    def __rsub__(self, other):
        if isinstance(other, Tensor):
            return other - self
        else:
            return other - self
        
    def __rtruediv__(self, other):
        if isinstance(other, Tensor):
            return other / self
        else:
            return other / self

    # representation of the Tensor
    def __repr__(self):
        if not isinstance(self.data, list):
            return f"Tensor({self.data})"
        if len(self.data) == 1 and not isinstance(self.data[0], list):
            return f"Tensor({self.data[0]})"
        return f"Tensor({self.data})"

    def __iter__(self):
        if not isinstance(self.data, list):
            raise TypeError("0-D Tensor is not iterable")
        if len(self.data) == 1 and not isinstance(self.data[0], list):
            raise TypeError("0-D Tensor is not iterable")
        for item in self.data:
            yield Tensor(item)
        
    def get_value(self):
        return self.data[0] if len(self.data) == 1 else self.data