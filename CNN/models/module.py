from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Union, Callable, Optional, Set, Iterator
import math
from torch_lite import Tensor, Parameter
from functional import sigmoid, relu
from random_lite import SimpleRandom
from dataclasses import dataclass

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

@dataclass
class ModelConfig:
    """
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int): Size of the convolutional kernel
        stride (int): Stride of the convolution
        padding (int): Padding added to both sides of the input
    """
    in_channels: int
    out_channels: int
    kernel_size: int
    stride: int = 1
    padding: int = 1
    

class Convol2D(Module):
    """2D Convolutional layer"""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.in_channels = config.in_channels
        self.out_channels = config.out_channels
        self.kernel_size = config.kernel_size
        self.stride = config.stride
        self.padding = config.padding

        # Calculate total elements needed
        total_elements = config.out_channels * config.in_channels * config.kernel_size * config.kernel_size

        std = math.sqrt(2.0 / (config.in_channels * config.kernel_size * config.kernel_size)) # recommended initialization for Conv2D layers I read somewhere
        flat_weights = [rng.normal(0, std) for _ in range(total_elements)]
        weights_tensor = Tensor(flat_weights)
        weights_reshaped = weights_tensor.reshape((config.out_channels, config.in_channels, config.kernel_size, config.kernel_size))
        
        self.weights = Parameter(weights_reshaped.data)
        self.biases = Parameter([rng.uniform(-0.5, 0.5) for _ in range(config.out_channels)]) 

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): Input tensor of shape (batch_size, in_channels, height, width)

        Returns:
            Output tensor with shape (batch_size, out_channels, height_out, width_out)
        """
        batch_size, in_channels, height, width = x.shape

        # Padding if needed
        if self.padding > 0:
            pad_batch = batch_size
            pad_in_channels = in_channels
            pad_height = height + 2 * self.padding
            pad_width = width + 2 * self.padding
            pad_im = Tensor.zeros((pad_batch, pad_in_channels, pad_height, pad_width))

            pad_im[:, :, self.padding:height + self.padding, self.padding:width + self.padding] += x # padded image
            x = pad_im
            height, width = pad_height, pad_width

        height_out = (height - self.kernel_size) // self.stride + 1
        width_out = (width - self.kernel_size) // self.stride + 1

        patches = self._im2col(x)  # Extract patches that will be used for convolution
        
        # (out_channels, in_channels, kernel_size, kernel_size) -> (out_channels, patch_size)
        patch_size = self.in_channels * self.kernel_size * self.kernel_size
        weights_2d = self.weights.reshape((self.out_channels, patch_size))

        # TODO: Check error if size doesn't match here
        # (out_channels, patch_size) @ (batch_size, patch_size, num_patches) -> (batch_size, out_channels, num_patches)
        conv_result = weights_2d @ patches

        bias_expanded = self.biases.reshape((1, self.out_channels, 1))
        output = conv_result + bias_expanded

        return output.reshape((batch_size, self.out_channels, height_out, width_out))

    def _im2col(self, x: Tensor) -> Tensor:
        """
        Use im2col to extract patches that will be used for convolution.
        Args:
            x (Tensor): Input tensor of shape (batch_size, in_channels, height, width)
        Returns:
            Tensor: Patches of shape (batch_size, in_channels * kernel_size * kernel_size, num_patches)
        """
        batch_size, in_channels, height, width = x.shape
        kernel_size = self.kernel_size
        stride = self.stride

        # Calculate output dimensions
        height_out = (height - kernel_size) // stride + 1
        width_out = (width - kernel_size) // stride + 1

        # Create an empty tensor to hold the patches
        num_patches = height_out * width_out
        patches = Tensor.zeros((batch_size, in_channels * kernel_size * kernel_size, num_patches))

        for i in range(height_out):
            for j in range(width_out):
                h_start = i * stride
                w_start = j * stride
                patch = x[:, :, h_start:h_start + kernel_size, w_start:w_start + kernel_size]
                patches[:, :, i * width_out + j] = patch.reshape(batch_size, -1)

        return patches # (batch_size, in_channels * kernel_size * kernel_size, num_patches)


class MaxPooling(Module):
    """Max pooling layer"""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.kernel_size = config.kernel_size
        self.stride = config.stride

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape (batch_size, in_channels, height, width)

        Returns:
            Output tensor with shape (batch_size, in_channels, height_out, width_out)
        """
        batch_size, in_channels, height, width = x.shape
        height_out = (height - self.kernel_size) // self.stride + 1
        width_out = (width - self.kernel_size) // self.stride + 1

        output = Tensor.zeros((batch_size, in_channels, height_out, width_out))

        for i in range(height_out):
            for j in range(width_out):
                h_start = i * self.stride
                w_start = j * self.stride
                patch = x[:, :, h_start:h_start + self.kernel_size, w_start:w_start + self.kernel_size]
                output[:, :, i, j] = patch.data.max(axis=(2, 3))

        return output


class Flatten(Module):
    """Flatten layer"""
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): Input for this (batch_size, in_channels, height_out, width_out)
        
        Returns:
            Tensor: Flattened tensor of shape (batch_size, flattened_features)
        """
        batch_size = x.shape[0]
        
        # Calculate total number of features per sample
        total_features = 1
        for dim in x.shape[1:]:  # Skip batch
            total_features *= dim
        
        # Flatten each sample in the batch
        flattened_data = []
        for i in range(batch_size):
            sample = x[i]  # Get individual sample
            flat_sample = sample.flatten()  # Flatten the sample
            flattened_data.append(flat_sample.data) # Append for each batch sample
        
        return Tensor(flattened_data, requires_grad=x.requires_grad)


class Linear(Module):
    """Linear/Dense/FC layer"""
    def __init__(self, in_features: int, out_features: int, bias:bool=True):
        """
        Args:
            in_features (int): Number of input features
            out_features (int): Number of output features
            bias (bool): Use bias or not
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        
        std = math.sqrt(2.0 / (in_features + out_features))

        weight_data = []
        for i in range(out_features):
            row = [rng.normal(0, std) for _ in range(in_features)]
            weight_data.append(row)
        
        self.weights = Parameter(weight_data)

        bias_data = [rng.uniform(-0.1, 0.1) for _ in range(out_features)]
        self.bias = Parameter(bias_data)

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape (batch_size, in_features)

        Returns:
            Output tensor with shape (batch_size, out_features)
        """
        batch_size, in_features = x.shape
        assert in_features == self.in_features, f"Input features {in_features} do not match layer's in_features {self.in_features}"
        
        output = x @ self.weights.transpose() # (batch_size, in_features) @ (in_features, out_features) -> (batch_size, out_features)

        if self.bias:
            output = output + self.bias

        return output


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