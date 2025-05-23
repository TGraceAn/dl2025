from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Union, Callable, Optional, Set, Iterator
import math

Num = Union[int, float]

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
    def __init__(self, data: Union[Num, List[Num]], requires_grad: bool=True):
        """
        Arg:
            data (Union[Num, List[Num]]): Data to store
            requires_grad (bool): If True, the tensor will track gradients
        """
    
        self.data: Union[Num, List[Num]] = data
        self.requires_grad: bool = requires_grad
        self.grad: Optional[float] = 0.0 if requires_grad else None
        self._backward: Callable[[], None] = lambda: None
        self._prev: Set[Tensor] = set()

    def __getitem__(self, index: int) -> Tensor:
        value = self.data[index]
        return Tensor(value) if not isinstance(value, list) else Tensor(value)

    # representation of the Tensor
    def __repr__(self) -> str:
        if isinstance(self.data, list):
            return f"Tensor({self.data})"
        else:
            return f"Tensor({self.data})"

    def __iter__(self) -> Iterator[Tensor]:
        if not isinstance(self.data, list):
            raise TypeError("0-D Tensor is not iterable")
        if len(self.data) == 1 and not isinstance(self.data[0], list):
            raise TypeError("0-D Tensor is not iterable")
        for item in self.data:
            yield Tensor(item)
        
    def get_value(self) -> Union[Num, List[Num]]:
        return self.data[0] if len(self.data) == 1 else self.data
    
    def backward(self):
        """Compute gradients"""
        if not self.requires_grad:
            raise RuntimeError("Tensor does not require gradients") # TODO: Build try/except
        
        # Build topological order
        topo = []
        visited = set()

        def build_topo(t):
            if t not in visited:
                visited.add(t)
                for child in t._prev:
                    build_topo(child)
                topo.append(t)

        build_topo(self)

        self.grad = 1.0
        
        # Backpropagate
        for node in reversed(topo):
            node._backward()

    ## Operators for tensor operations ## 
    def __add__(self, other: Union[Num, Tensor]) -> Tensor:
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False)
        
        out = Tensor(self.data + other.data, requires_grad=self.requires_grad or other.requires_grad)
        out._prev = {self, other}

        def _backward():
            if self.requires_grad:
                self.grad += out.grad
            if other.requires_grad:
                other.grad += out.grad
        
        out._backward = _backward
        return out
        
    def __sub__(self, other: Union[Num, Tensor]) -> Tensor:
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False)
        
        out = Tensor(self.data - other.data, requires_grad=self.requires_grad or other.requires_grad)
        out._prev = {self, other}

        def _backward():
            if self.requires_grad:
                self.grad += out.grad
            if other.requires_grad:
                other.grad -= out.grad
        
        out._backward = _backward
        return out
        
    def __mul__(self, other: Union[Num, Tensor]) -> Tensor:
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False)
        
        out = Tensor(self.data * other.data, requires_grad=self.requires_grad or other.requires_grad)
        out._prev = {self, other}

        def _backward():
            if self.requires_grad:
                self.grad += other.data * out.grad
            if other.requires_grad:
                other.grad += self.data * out.grad
        
        out._backward = _backward
        return out
        
    def __truediv__(self, other : Union[Num, Tensor]) -> Tensor:
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False)
        
        # Check for division by zero
        if other.data == 0:
            raise ZeroDivisionError("Division by zero in tensor operation")
        
        out = Tensor(self.data / other.data, requires_grad=self.requires_grad or other.requires_grad)
        out._prev = {self, other}

        def _backward():
            if self.requires_grad:
                self.grad += out.grad / other.data
            if other.requires_grad:
                other.grad -= out.grad * self.data / (other.data ** 2)
        
        out._backward = _backward
        return out

    def __pow__(self, other : Union[Num, Tensor]) -> Tensor:
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False)
        
        # Check for problematic cases
        if self.data == 0 and other.data < 0:
            raise ZeroDivisionError("Cannot raise 0 to a negative power")
        if self.data < 0 and not isinstance(other.data, int):
            raise ValueError("Cannot raise negative number to non-integer power")
        if self.data == 0 and other.data == 0:
            raise ValueError("0^0 is undefined")
        
        out = Tensor(self.data ** other.data, requires_grad=self.requires_grad or other.requires_grad)
        out._prev = {self, other}

        def _backward():
            if self.requires_grad:
                self.grad += out.grad * other.data * (self.data ** (other.data - 1))
            if other.requires_grad:
                # Only compute log gradient if self.data > 0
                if self.data > 0:
                    other.grad += out.grad * math.log(self.data) * (self.data ** other.data)
                else:
                    # For self.data <= 0, the log term is undefined; set gradient to 0
                    other.grad += 0
        
        out._backward = _backward
        return out
        
    def __neg__(self) -> Tensor:
        out = Tensor(-self.data, requires_grad=self.requires_grad)
        out._prev = {self}

        def _backward():
            if self.requires_grad:
                self.grad -= out.grad
        
        out._backward = _backward
        return out
    
    def __radd__(self, other: Union[Num, Tensor]) -> Tensor:
        return self + other

    def __rsub__(self, other: Union[Num, Tensor]) -> Tensor:
        return Tensor(other, requires_grad=False) - self

    def __rmul__(self, other: Union[Num, Tensor]) -> Tensor:
        return self * other

    def __rtruediv__(self, other: Union[Num, Tensor]) -> Tensor:
        other_tensor = Tensor(other, requires_grad=False)
        return other_tensor / self