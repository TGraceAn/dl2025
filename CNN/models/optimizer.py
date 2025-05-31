from abc import ABC, abstractmethod
from module import Module
from typing import List, Optional, Dict, Union
from torch_lite import Parameter

class Optimizer(ABC):
    """Optimizer for the module"""
    def __init__(self, model: Module, lr: float):
        """Initialize the optimizer for a given model."""
        self.model = model
        self.lr = lr
        self.params = model.parameters()

    @abstractmethod
    def update_param(self, param: Parameter):
        """Update a single parameter"""

    def step(self):
        """Update all parameters in the model"""
        for param in self.params:
            if param is not None and param.requires_grad:
                self.update_param(param)

    def zero_grad(self):
        """Zero the gradients of all parameters"""
        for param in self.params:
            if param is not None:
                param.zero_grad()
    
    
class SGD(Optimizer):
    """Stochastic Gradient Descent"""
    
    def __init__(self, model: Module, lr: float):
        """
        Initialize simple SGD optimizer
        
        Args:
            model (Module): The model to optimize
            lr (float): Learning rate
        """
        super().__init__(model, lr)
    
    def update_param(self, param: Parameter):
        """
        Updated version that properly modifies parameter data in place
        
        Args:
            param (Parameter): Parameter to update
        """
        if param.grad is None:
            return
        
        # Create updated data similar to the original data
        def compute_update(data, grad):
            if isinstance(data, list) and isinstance(grad, list):
                return [compute_update(d, g) for d, g in zip(data, grad)]
            else:
                return data - self.lr * grad
        
        # Update the version
        param.data = compute_update(param.data, param.grad)

