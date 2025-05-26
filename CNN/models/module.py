from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Union, Callable, Optional, Set, Iterator
import math
from torch_lite import Tensor, Parameter
from functional import sigmoid, relu
from random_lite import SimpleRandom

rng = SimpleRandom(seed=11) 

"""Initial stuff for CNN"""
class Module(ABC):
    """Interface for all modules"""

    @abstractmethod
    def __init__(self):
        """Initialize the module"""

    @abstractmethod
    def forward(self, x):
        """Forward pass"""
        
    @abstractmethod
    def backward(self, grad_output):
        """Backward pass"""

    @abstractmethod
    def step(self, lr):
        """Update the weights"""

    @abstractmethod
    def zero_grad(self):
        """Zero the gradients"""

    def __call__(self, x):
        """Call the forward method"""
        return self.forward(x)
        
    ## Lol I overthinked so I'm just gonna note here to remmber if someday I want to implement an option with no bias
    def zero_grad(self):
        """Zero the gradients of the module"""
        # For every weight and bias in the layer, set the gradient to zero
        # Get all children of the module

    # # TODO: Implement freeze and unfreeze methods for fine-tuning or transfer learning later
    # def freeze(self):
    #     """Freeze the weights and biases of the layer"""
    #     self.weights.requires_grad = False
    #     self.biases.requires_grad = False

    # def unfreeze(self):
    #     """Unfreeze the weights and biases of the layer"""
    #     self.weights.requires_grad = True
    #     self.biases.requires_grad = True

    def step(self, lr: float):
        """Update the weights using the optimizer"""
        pass


class Convol2D(Module):
    """2D Convolutional layer"""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0):
        """
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            kernel_size (int): Size of the convolutional kernel
            stride (int): Stride of the convolution
            padding (int): Padding added to both sides of the input
            requires_grad (bool): If True, gradients will be computed for this layer 
                (False use for inference or for frozen layers later for fine-tuning or transfer learning)
        """

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        std = math.sqrt(2.0 / (in_channels * kernel_size * kernel_size)) # recommended initialization for Conv2D layers I read somewhere
        weights_data = [[[[rng.normal(0, std) for _ in range(kernel_size)] 
                          for _ in range(kernel_size)] 
                         for _ in range(in_channels)] 
                        for _ in range(out_channels)]
        self.weights = Parameter(weights_data) 
        self.biases = Parameter([rng.uniform(-0.5, 0.5) for _ in range(out_channels)]) 
        self.last_input = None

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): Input tensor of shape (batch_size, in_channels, height, width)

        Returns:
            Output tensor with shape (batch_size, out_channels, height_out, width_out)
        """
        self.last_input = x
        batch_size, _, height, width = x.shape

        if self.padding > 0:
            x_padded = Tensor.zeros((batch_size, self.in_channels,
                                     height + 2 * self.padding,
                                     width + 2 * self.padding), requires_grad=x.requires_grad)
            for b in range(batch_size):
                for c in range(self.in_channels):
                    for i in range(height):
                        for j in range(width):
                            x_padded.data[b][c][i + self.padding][j + self.padding] = x.data[b][c][i][j]
        else:
            x_padded = x

        padded_height = x_padded.shape[2]
        padded_width = x_padded.shape[3]

        height_out = (padded_height - self.kernel_size) // self.stride + 1
        width_out = (padded_width - self.kernel_size) // self.stride + 1

        output_list = []
        for b in range(batch_size):
            batch_list = []
            for out_c in range(self.out_channels):
                channel_list = []
                for i in range(height_out):
                    row_list = []
                    for j in range(width_out):
                        h_start = i * self.stride
                        w_start = j * self.stride
                        h_end = h_start + self.kernel_size
                        w_end = w_start + self.kernel_size

                        conv_sum = Parameter(0.0)

                        for in_c in range(self.in_channels):
                            region = x_padded[b, in_c, h_start:h_end, w_start:w_end]
                            kernel = Tensor(self.weights.data[out_c][in_c], requires_grad=self.weights.requires_grad)
                            conv_sum = conv_sum + (region * kernel).sum()

                        result = conv_sum + self.biases[out_c]
                        row_list.append(result.data)
                    channel_list.append(row_list)
                batch_list.append(channel_list)
            output_list.append(batch_list)

        requires_grad = x.requires_grad or self.weights.requires_grad or self.biases.requires_grad
        return Parameter(output_list, requires_grad=requires_grad)

    def step(self, lr):
        pass

    def __call__(self, x):
        return super().__call__(x)


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
    
class SGD(Optimizer):
    ...

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
    

# TODO: after eveything is done, make a init weights function