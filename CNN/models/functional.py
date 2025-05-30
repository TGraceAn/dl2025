from __future__ import annotations
import math
from torch_lite import Tensor, Num, _add_gradients, _elementwise_op
from typing import Union

def sigmoid(x: Union[Tensor, Num]) -> Tensor:
    """
    Sigmoid activation function: 1 / (1 + exp(-x))
    
    Args:
        x: Input tensor or number

    Returns:
        Tensor with sigmoid applied
    """
    if not isinstance(x, Tensor):
        x = Tensor(x)
    
    def _sigmoid(val):
        if isinstance(val, list):
            return [_sigmoid(item) for item in val]
        else:
            return 1.0 / (1.0 + math.exp(-val))
    
    sigmoid_val = _sigmoid(x.data)
    
    out = Tensor(sigmoid_val, requires_grad=x.requires_grad) 
    out._prev = {x}
    
    def _backward():
        # Gradient of sigmoid: sigmoid(x) * (1 - sigmoid(x))
        def _sigmoid_grad(sig_val):
            if isinstance(sig_val, list):
                return [_sigmoid_grad(item) for item in sig_val]
            else:
                return sig_val * (1.0 - sig_val)
        
        sigmoid_grad = _sigmoid_grad(sigmoid_val)
        grad_contribution = _elementwise_op(out.grad, sigmoid_grad, lambda x, y: x * y)
        x.grad = _add_gradients(x.grad, grad_contribution)
    
    out._backward = _backward
    return out


def relu(x: Union[Tensor, Num]) -> Tensor:
    """
    ReLU activation function: max(0, x)
    
    Args:
        x: Input tensor or number
        
    Returns:
        Tensor with ReLU applied
    """
    if not isinstance(x, Tensor):
        x = Tensor(x)
    
    def _relu(val):
        if isinstance(val, list):
            return [_relu(item) for item in val]
        else:
            return max(0.0, val)
    
    relu_val = _relu(x.data)
    
    out = Tensor(relu_val, requires_grad=x.requires_grad)
    out._prev = {x}
    
    def _backward():
        # Gradient of ReLU: 1 if x > 0 else 0
        def _relu_grad(orig_val):
            if isinstance(orig_val, list):
                return [_relu_grad(item) for item in orig_val]
            else:
                return 1.0 if orig_val > 0 else 0.0
        
        relu_grad = _relu_grad(x.data)
        grad_contribution = _elementwise_op(out.grad, relu_grad, lambda x, y: x * y)
        x.grad = _add_gradients(x.grad, grad_contribution)
    
    out._backward = _backward
    return out


def tanh(x: Union[Tensor, Num]) -> Tensor:
    """
    Tanh activation function: (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    
    Args:
        x: Input tensor or number
        
    Returns:
        Tensor with tanh applied
    """
    if not isinstance(x, Tensor):
        x = Tensor(x, requires_grad=False)
    
    def _tanh(val):
        if isinstance(val, list):
            return [_tanh(item) for item in val]
        else:
            return math.tanh(val)
    
    tanh_val = _tanh(x.data)
    
    out = Tensor(tanh_val, requires_grad=x.requires_grad)
    out._prev = {x}
    
    def _backward():
        def _tanh_grad(tanh_result):
            if isinstance(tanh_result, list):
                return [_tanh_grad(item) for item in tanh_result]
            else:
                return 1.0 - tanh_result ** 2
        
        tanh_grad = _tanh_grad(tanh_val)
        grad_contribution = _elementwise_op(out.grad, tanh_grad, lambda x, y: x * y)
        x.grad = _add_gradients(x.grad, grad_contribution)
    
    out._backward = _backward
    return out