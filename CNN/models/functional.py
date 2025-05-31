from __future__ import annotations
import math
from torch_lite import Tensor, Num, _add_gradients, _elementwise_op
from typing import Union
import math

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

def softmax(x: Union[Tensor, Num], dim: int = -1) -> Tensor:
    """
    Args:
        x: Input tensor
        dim: Dimension to apply softmax (default: last dimension)
        
    Returns:
        Tensor 
    """
    if not isinstance(x, Tensor):
        x = Tensor(x)
    
    def _softmax_1d(logits_vec):
        """Apply softmax to a vector"""
        if not isinstance(logits_vec, list):
            return 1.0
        
        max_logit = max(logits_vec)
        shifted = [val - max_logit for val in logits_vec]
        
        # Compute exp and sum
        exp_vals = [math.exp(val) for val in shifted]
        sum_exp = sum(exp_vals)
        
        # Return probabilities
        return [val / sum_exp for val in exp_vals]
    
    def _apply_softmax_recursive(data, current_dim, target_dim, shape):
        if current_dim == target_dim:
            return _softmax_1d(data)
        else:
            return [_apply_softmax_recursive(item, current_dim + 1, target_dim, shape) for item in data]
    
    if dim < 0:
        dim = len(x.shape) + dim
    
    if dim >= len(x.shape):
        raise ValueError(f"Dimension {dim} is out of bounds for tensor of shape {x.shape}")
    
    softmax_data = _apply_softmax_recursive(x.data, 0, dim, x.shape)
    
    out = Tensor(softmax_data, requires_grad=x.requires_grad)
    out._prev = {x}
    
    def _backward():
        def _softmax_grad_1d(softmax_vals, grad_vals):
            """Compute gradient"""
            if not isinstance(softmax_vals, list):
                return grad_vals
            
            n = len(softmax_vals)
            grad = [0.0] * n
            
            for i in range(n):
                for j in range(n):
                    if i == j:
                        grad[i] += grad_vals[j] * softmax_vals[i] * (1 - softmax_vals[i])
                    else:
                        grad[i] -= grad_vals[j] * softmax_vals[i] * softmax_vals[j]
            
            return grad
        
        def _apply_softmax_grad_recursive(softmax_data, grad_data, current_dim, target_dim):
            if current_dim == target_dim:
                return _softmax_grad_1d(softmax_data, grad_data)
            else:
                return [_apply_softmax_grad_recursive(s, g, current_dim + 1, target_dim) 
                       for s, g in zip(softmax_data, grad_data)]
        
        grad_contribution = _apply_softmax_grad_recursive(softmax_data, out.grad, 0, dim)
        x.grad = _add_gradients(x.grad, grad_contribution)
    
    out._backward = _backward
    return out

def cross_entropy(logits: Tensor, targets: Tensor) -> Tensor:
    """
    Args:
        logits: Tensor (batch_size, num_classes)
        targets: Tensor (batch_size,)

    Returns:
        Scalar Tensor representing the mean cross entropy loss for the batch
    """
    if not isinstance(logits, Tensor):
        logits = Tensor(logits)
    if not isinstance(targets, Tensor):
        targets = Tensor(targets)
    
    def _cross_entropy_batch(logits_batch, targets_batch):
        """Compute cross entropy for a batch"""
        losses = []
        for logits_sample, target_idx in zip(logits_batch, targets_batch):
            if isinstance(target_idx, float):
                target_idx = int(target_idx)

            # Numerically stable softmax + cross entropy
            max_logit = max(logits_sample)
            shifted_logits = [x - max_logit for x in logits_sample]
            sum_exp = sum(math.exp(x) for x in shifted_logits)
            log_sum_exp = math.log(sum_exp)

            # Cross entropy: -log(softmax[target]) = -(logit[target] - log_sum_exp)
            loss = -(shifted_logits[target_idx] - log_sum_exp)
            losses.append(loss)
        return losses

    # Compute mean loss
    losses = _cross_entropy_batch(logits.data, targets.data)
    loss_val = sum(losses) / len(losses)
    
    out = Tensor(loss_val, requires_grad=logits.requires_grad)
    out._prev = {logits}
    
    def _backward():
        batch_size = len(logits.data)
        grad_batch = []

        for logits_sample, target_idx in zip(logits.data, targets.data):
            if isinstance(target_idx, float):
                target_idx = int(target_idx)

            max_logit = max(logits_sample)
            shifted_logits = [x - max_logit for x in logits_sample]
            exp_vals = [math.exp(x) for x in shifted_logits]
            sum_exp = sum(exp_vals)
            softmax_vals = [x / sum_exp for x in exp_vals]

            grad_sample = softmax_vals.copy()
            grad_sample[target_idx] -= 1.0
            grad_sample = [x / batch_size for x in grad_sample]

            grad_batch.append(grad_sample)

        if isinstance(out.grad, (int, float)):
            final_grad = [[out.grad * x for x in sample] for sample in grad_batch]
        else:
            final_grad = grad_batch 

        logits.grad = _add_gradients(logits.grad, final_grad)

    out._backward = _backward
    return out
