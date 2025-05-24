from __future__ import annotations
from typing import List, Union, Callable, Optional, Set, Iterator
import math

Num = Union[int, float]

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
        if not self.requires_grad: # Too lazy to check if the tensor is a leaf node
            return

        
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
                if self.grad is None:
                    self.grad = 0.0
                self.grad += out.grad
            if other.requires_grad:
                if other.grad is None:
                    other.grad = 0.0
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
                if self.grad is None:
                    self.grad = 0.0
                self.grad += out.grad
            if other.requires_grad:
                if other.grad is None:
                    other.grad = 0.0
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
                if self.grad is None:
                    self.grad = 0.0
                self.grad += other.data * out.grad
            if other.requires_grad:
                if other.grad is None:
                    other.grad = 0.0
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
                if self.grad is None:
                    self.grad = 0.0
                self.grad += out.grad / other.data
            if other.requires_grad:
                if other.grad is None:
                    other.grad = 0.0
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
                if self.grad is None:
                    self.grad = 0.0
                self.grad += out.grad * other.data * (self.data ** (other.data - 1))
            if other.requires_grad:
                if other.grad is None:
                    other.grad = 0.0
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
                if self.grad is None:
                    self.grad = 0.0
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

    @property
    def shape(self):
        return (1,) if isinstance(self.data, (int, float)) else (len(self.data),)
    
    def detach(self) -> Tensor:
        return Tensor(self.data, requires_grad=False)

    def clone(self) -> Tensor:
        return Tensor(self.data, requires_grad=self.requires_grad)

    # # Maybe use -> casue I might implement it somewhere else
    # def zero_grad(self):
    #     visited = set()
    #     def _zero(t):
    #         if t not in visited:
    #             visited.add(t)
    #             if t.requires_grad:
    #                 t.grad = 0.0
    #             for child in t._prev:
    #                 _zero(child)
    #     _zero(self)