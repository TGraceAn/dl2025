from __future__ import annotations
import math
from torch_lite import Tensor, Num
from typing import Union


def sigmoid(x: Union[Tensor, Num], requires_grad: bool = False) -> Tensor:
    """    
    Args:
        x: Input tensor or number

    Returns:
        Tensor with sigmoid applied
    """
    if not isinstance(x, Tensor):
        x = Tensor(x, requires_grad=requires_grad)
    
    sigmoid_val = 1.0 / (1.0 + math.exp(-x.data))
    
    out = Tensor(sigmoid_val, requires_grad=x.requires_grad)
    out._prev = {x}
    
    def _backward():
        if x.requires_grad:
            # Gradient of sigmoid: sigmoid(x) * (1 - sigmoid(x))
            sigmoid_grad = sigmoid_val * (1.0 - sigmoid_val)
            x.grad += out.grad * sigmoid_grad
    
    out._backward = _backward
    return out

def relu(x: Union[Tensor, Num], requires_grad: bool = False) -> Tensor:
    """
    ReLU activation function:
    
    Args:
        x: Input tensor or number
        
    Returns:
        Tensor with ReLU applied
    """
    if not isinstance(x, Tensor):
        x = Tensor(x, requires_grad=requires_grad) # just a value, no need to track gradients
    
    relu_val = max(0.0, x.data)
    
    out = Tensor(relu_val, requires_grad=x.requires_grad)
    out._prev = {x}
    
    def _backward():
        if x.requires_grad:
            # Gradient of ReLU: 1 if x > 0 else 0
            relu_grad = 1.0 if x.data > 0 else 0.0
            x.grad += out.grad * relu_grad
    
    out._backward = _backward
    return out

# Extra cause this is the first activation I learnt when I first learnt nn coding
def tanh(x: Union[Tensor, Num], requires_grad: bool = False) -> Tensor:
    """    
    Args:
        x: Input tensor or number
        
    Returns:
        Tensor with tanh applied
    """
    if not isinstance(x, Tensor):
        x = Tensor(x, requires_grad=requires_grad)
    
    # Compute tanh value
    tanh_val = math.tanh(x.data)
    
    out = Tensor(tanh_val, requires_grad=x.requires_grad)
    out._prev = {x}
    
    def _backward():
        if x.requires_grad:
            # Gradient of tanh: 1 - tanh^2(x)
            tanh_grad = 1.0 - tanh_val ** 2
            x.grad += out.grad * tanh_grad
    
    out._backward = _backward
    return out