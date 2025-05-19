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