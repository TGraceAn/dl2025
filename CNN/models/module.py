from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Union, Callable, Optional, Set, Iterator
import math
from torch_lite import Tensor, Parameter, _add_gradients
from functional import sigmoid, relu
from random_lite import SimpleRandom
from dataclasses import dataclass

rng = SimpleRandom(seed=11) 

"""Initial stuff for CNN"""
class Module(ABC):
    """Interface for all modules"""
    def __init__(self):
        """Initialize the module"""
        self.weights: Optional[Parameter] = None
        self.biases: Optional[Parameter] = None

    @abstractmethod
    def forward(self, x):
        """Forward pass"""

    def parameters(self) -> List[Parameter]:
        """Return all parameters of the module and its submodules"""
        params = []
        
        for attr_name in dir(self):
            if not attr_name.startswith('_'):  # Skip private attributes
                attr = getattr(self, attr_name)
                
                if isinstance(attr, Parameter):
                    params.append(attr)
                elif isinstance(attr, Module) and attr is not self:

                    params.extend(attr.parameters())

                # # For things that's similar to nn.ModuleList 
                # elif isinstance(attr, (list, tuple)):
                #     for item in attr:
                #         if isinstance(item, Parameter):
                #             params.append(item)
                #         elif isinstance(item, Module):
                #             params.extend(item.parameters())

        
        return params


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

# @dataclass
# class ModelConfig:
#     """
#     Args:
#         in_channels (int): Number of input channels
#         out_channels (int): Number of output channels
#         kernel_size (int): Size of the convolutional kernel
#         stride (int): Stride of the convolution
#         padding (int): Padding added to both sides of the input
#     """
#     in_channels: int
#     out_channels: int
#     kernel_size: int
#     stride: int = 1
#     padding: int = 1
    

class Convol2D(Module):
    """2D Convolutional layer"""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Calculate total elements needed
        total_elements = out_channels * in_channels * kernel_size * kernel_size

        std = math.sqrt(2.0 / (in_channels * kernel_size * kernel_size)) # recommended initialization for Conv2D layers I read somewhere
        flat_weights = [rng.normal(0, std) for _ in range(total_elements)]
        weights_tensor = Tensor(flat_weights)
        
        weights_reshaped = weights_tensor.reshape((out_channels, in_channels, kernel_size, kernel_size))

        self.weights = Parameter(weights_reshaped.data)
        self.biases = Parameter([rng.uniform(-0.5, 0.5) for _ in range(out_channels)]) 

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
        # (batch_size, patch_size, num_patches)
        
        # (out_channels, in_channels, kernel_size, kernel_size) -> (out_channels, patch_size)
        patch_size = self.in_channels * self.kernel_size * self.kernel_size
        weights_2d = self.weights.reshape((self.out_channels, patch_size))

        # TODO: Check error if size doesn't match here
        # (out_channels, patch_size) @ (batch_size, patch_size, num_patches) -> (batch_size, out_channels, num_patches)
        conv_result = weights_2d @ patches
        
        # Create bias tensor with proper broadcasting shape
        for c in range(self.out_channels):
            conv_result[:, c, :] += self.biases[c]

        output = conv_result

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

                patch_size = in_channels * kernel_size * kernel_size
                patches[:, :, i * width_out + j] = patch.reshape((batch_size, patch_size))

        return patches # (batch_size, in_channels * kernel_size * kernel_size, num_patches)


class MaxPooling(Module):
    """Max pooling layer"""
    def __init__(self, kernel_size: int, stride: int = 2):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

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
        output.requires_grad = x.requires_grad  # requires_grad from input

        for i in range(height_out):
            for j in range(width_out):
                h_start = i * self.stride
                w_start = j * self.stride
                patch = x[:, :, h_start:h_start + self.kernel_size, w_start:w_start + self.kernel_size]

                max_vals = patch.max(axis=(2, 3))
                output[:, :, i, j] = max_vals.data  

        output._prev = {x}
        output._backward = lambda: setattr(x, 'grad', _add_gradients(x.grad, output.grad)) if output.grad else None
        
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
        
        return x.reshape((batch_size, total_features))


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
        self.biases = Parameter(bias_data)

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

        if self.bias and self.biases is not None:
            for c in range(self.out_channels):
                output[:, c] += self.biases[c]

        return output