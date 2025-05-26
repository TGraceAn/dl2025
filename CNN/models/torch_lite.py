from __future__ import annotations
from typing import List, Union, Callable, Optional, Set, Iterator, Tuple
import math

Num = Union[int, float]

# Helper functions
def _elementwise_op(a, b, op):
    if isinstance(a, list) and isinstance(b, list):
        return [_elementwise_op(x, y, op) for x, y in zip(a, b)]
    elif isinstance(a, list):
        return [_elementwise_op(x, b, op) for x in a]
    elif isinstance(b, list):
        return [_elementwise_op(a, y, op) for y in b]
    else:
        return op(a, b)
    
def _add_gradients(grad1, grad2):
    """Add two gradients with same structure"""
    if grad1 is None:
        return grad2
    if grad2 is None:
        return grad1
    return _elementwise_op(grad1, grad2, lambda x, y: x + y)
    
# Create zeros with same structure as data
def _zeros_like(data):
    """Create zeros with same structure as data"""
    if isinstance(data, list):
        return [_zeros_like(item) for item in data]
    else:
        return 0.0

# Create ones with same structure as data
def _ones_like(data):
    """Create ones with same structure as data"""
    if isinstance(data, list):
        return [_ones_like(item) for item in data]
    else:
        return 1.0

def _sum_recursive(data):
    """Recursively sum all elements"""
    if isinstance(data, list):
        return sum(_sum_recursive(item) for item in data)
    else:
        return data


class Tensor:
    """Tensor class"""
    def __init__(self, data: Union[Num, List[Num]], requires_grad: bool=False):
        """
        Arg:
            data (Union[Num, List[Num]]): Data to store
            requires_grad (bool): If False, the tensor will track gradients
        """
    
        self.data: Union[Num, List[Num]] = data
        self.requires_grad: bool = requires_grad
        self.grad: Optional[Union[Num, List[Num]]] = _zeros_like(data) if requires_grad else None

    def __getitem__(self, index) -> Tensor:
        """Better indexing support"""
        def _get_item_recursive(data, indices):
            if not indices:
                return data
            if isinstance(indices[0], slice):
                start, stop, step = indices[0].indices(len(data))
                sliced_data = data[start:stop:step]
                if len(indices) == 1:
                    return sliced_data
                return [_get_item_recursive(item, indices[1:]) for item in sliced_data]
            else:
                if len(indices) == 1:
                    return data[indices[0]]
                return _get_item_recursive(data[indices[0]], indices[1:])
        
        if isinstance(index, tuple):
            value = _get_item_recursive(self.data, list(index))
        elif isinstance(index, slice):
            start, stop, step = index.indices(len(self.data))
            value = self.data[start:stop:step]
        else:
            value = self.data[index]
        
        return Tensor(value, requires_grad=self.requires_grad)

    # representation of the Tensor
    def __repr__(self) -> str:
        return f"Tensor({self.data})"

    def __iter__(self) -> Iterator[Tensor]:
        if not isinstance(self.data, list):
            raise TypeError("0-D Tensor is not iterable")
        for item in self.data:
            yield Tensor(item, requires_grad=self.requires_grad)
        
    def get_value(self) -> Union[Num, List[Num]]:
        return self.data[0] if len(self.data) == 1 else self.data
    
    def __add__(self, other: Union[Num, Tensor]) -> Tensor:
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False)
        data = _elementwise_op(self.data, other.data, lambda x, y: x + y)
        
        out = Tensor(data, requires_grad=self.requires_grad or other.requires_grad)
        return out
        
    def __sub__(self, other: Union[Num, Tensor]) -> Tensor:
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False)
        data = _elementwise_op(self.data, other.data, lambda x, y: x - y)

        out = Tensor(data, requires_grad=self.requires_grad or other.requires_grad)
        return out
        
    def __mul__(self, other: Union[Num, Tensor]) -> Tensor:
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False)
        data = _elementwise_op(self.data, other.data, lambda x, y: x * y)
        
        out = Tensor(data, requires_grad=self.requires_grad or other.requires_grad)
        return out
        
    def __truediv__(self, other : Union[Num, Tensor]) -> Tensor:
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False)
        
        # Check for division by zero
        if other.data == 0:
            raise ZeroDivisionError("Division by zero in tensor operation")
        data = _elementwise_op(self.data, other.data, lambda x, y: x / y)

        out = Tensor(data, requires_grad=self.requires_grad or other.requires_grad)
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
        
        data = _elementwise_op(self.data, other.data, lambda x, y: x ** y)
        out = Tensor(data, requires_grad=self.requires_grad or other.requires_grad)
        return out
        
    def __neg__(self) -> Tensor:
        out = Tensor(-self.data, requires_grad=self.requires_grad)
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

    def sum(self) -> Tensor:
        """Sum all elements in the tensor"""
        total = _sum_recursive(self.data)
        
        out = Tensor(total, requires_grad=self.requires_grad)
        return out

    def detach(self) -> Tensor:
        return Tensor(self.data, requires_grad=False)

    def clone(self) -> Tensor:
        return Tensor(self.data, requires_grad=self.requires_grad)

    def zero_grad(self):
        """Zero the gradients"""
        if self.requires_grad:
            self.grad = _zeros_like(self.data)
    
    @classmethod
    def zeros(cls, shape: Tuple[int, ...], requires_grad: bool = False) -> Tensor:
        """
        Create a tensor filled with zeros.
        Args: 
            shape (Tuple[int, ...]): Shape of the tensor to create
            requires_grad (bool): If True, the tensor will track gradients
        Returns:
            Tensor: A tensor filled with zeros of the specified shape
        """
        def build_zeros(s):
            if len(s) == 1:
                return [0.0] * s[0]
            return [build_zeros(s[1:]) for _ in range(s[0])]
        
        if len(shape) == 0:
            data = 0.0
        else:
            data = build_zeros(shape)
        
        return cls(data, requires_grad=requires_grad)
    
    @property
    def shape(self) -> Tuple[int, ...]:
        def _shape(data):
            if isinstance(data, list):
                if len(data) == 0:
                    return (0,)
                return (len(data),) + _shape(data[0])
            else:
                return ()
        return _shape(self.data)
    
    def flatten(self) -> Tensor:
        """
        Flatten tensor to 1D.
        
        Returns:
            New 1D tensor with flattened data
        """
        flat_data = self._flatten_recursive(self.data)
        return Tensor(flat_data)
    
    def reshape(self, new_shape: Tuple[int, ...]) -> Tensor:
        """
        Reshape tensor to new shape.
        Args:
            new_shape: Target shape as tuple of integers
            
        Returns:
            New tensor with reshaped data
        """

        current_elements = 1
        for dim in self.shape:
            current_elements *= dim

        new_elements = 1
        for dim in new_shape:
            new_elements *= dim
        
        if current_elements != new_elements:
            raise ValueError(f"Cannot reshape tensor of size {current_elements} to shape {new_shape} (size {new_elements})")
        
        flat_data = self._flatten_recursive(self.data)
        new_data = self._reshape_flat_to_nested(flat_data, new_shape)
        
        return Tensor(new_data)

    def _flatten_recursive(self, data):
        if isinstance(data, list):
            result = []
            for item in data:
                result.extend(self._flatten_recursive(item))
            return result
        else:
            return [data]

    def _reshape_flat_to_nested(self, flat_data: List[float], shape: Tuple[int, ...]) -> Union[float, List]:
        """Reshape flat data to nested structure based on shape."""
        if len(shape) == 0:
            return flat_data[0]
        elif len(shape) == 1:
            return flat_data[:shape[0]]
        else:
            # Calculate size of outer dimension
            outer_size = shape[0]
            inner_elements = 1
            for dim in shape[1:]:
                inner_elements *= dim
            
            result = []
            for i in range(outer_size):
                start_idx = i * inner_elements
                end_idx = start_idx + inner_elements
                inner_data = flat_data[start_idx:end_idx]
                inner_shape = shape[1:]
                result.append(self._reshape_flat_to_nested(inner_data, inner_shape))
            return result

    def transpose(self) -> Tensor:
        """ Transpose a 2D tensor. """
        if len(self.shape) != 2:
            raise ValueError("Transpose is only supported for 2D tensors.")
        
        rows, cols = self.shape
        transposed_data = [[self.data[j][i] for j in range(rows)] for i in range(cols)]
        return Tensor(transposed_data)

    def __matmul__(self, other: Tensor) -> Tensor:
        """ Matrix multiplication """
        if not isinstance(other, Tensor):
            other = Tensor(other)

        # Validate shapes
        if len(self.shape) != 2 or len(other.shape) != 2:
            raise ValueError("Matrix multiplication is only supported for 2D tensors.")
        if self.shape[1] != other.shape[0]:
            raise ValueError(f"Mismatched dimensions for matrix multiplication: {self.shape} and {other.shape}")

        # Perform multiplication
        result_data = [[sum(a * b for a, b in zip(A_row, B_col)) for B_col in zip(*other.data)] for A_row in self.data]
        
        return Tensor(result_data, requires_grad=self.requires_grad or other.requires_grad)


class Parameter(Tensor):
    """Parameter class that inherits from Tensor"""
    def __init__(self, data: Union[Num, List[Num]], requires_grad: bool = True):
        """
        Args:
            data (Union[Num, List[Num]]): Data to store
            requires_grad (bool): If True, the parameter will track gradients (default: True)
        """
        super().__init__(data, requires_grad=requires_grad)
        self._backward: Callable[[], None] = lambda: None
        self._prev: Set[Tensor] = set()
    
    def __repr__(self) -> str:
        return f"Parameter({self.data})"
    
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

        self.grad = _ones_like(self.data)
        
        # Backpropagate
        for node in reversed(topo):
            node._backward()

    ## Override operators to support gradient computation ##
    def __add__(self, other: Union[Num, Tensor]) -> Parameter:
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False)
        data = _elementwise_op(self.data, other.data, lambda x, y: x + y)
        
        out = Parameter(data, requires_grad=self.requires_grad or other.requires_grad)
        out._prev = {self, other}

        def _backward():
            if self.requires_grad and self.grad is not None:
                self.grad = _add_gradients(self.grad, out.grad)
            if other.requires_grad and other.grad is not None:
                other.grad = _add_gradients(other.grad, out.grad)
        
        out._backward = _backward
        return out
        
    def __sub__(self, other: Union[Num, Tensor]) -> Parameter:
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False)
        data = _elementwise_op(self.data, other.data, lambda x, y: x - y)

        out = Parameter(data, requires_grad=self.requires_grad or other.requires_grad)
        out._prev = {self, other}

        def _backward():
            if self.requires_grad and self.grad is not None:
                self.grad = _add_gradients(self.grad, out.grad)
            if other.requires_grad and other.grad is not None:
                neg_grad = _elementwise_op(out.grad, -1, lambda x, y: x * y)
                other.grad = _add_gradients(other.grad, neg_grad)
        
        out._backward = _backward
        return out
        
    def __mul__(self, other: Union[Num, Tensor]) -> Parameter:
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False)
        data = _elementwise_op(self.data, other.data, lambda x, y: x * y)
        
        out = Parameter(data, requires_grad=self.requires_grad or other.requires_grad)
        out._prev = {self, other}

        def _backward():
            if self.requires_grad and self.grad is not None:
                grad_contribution = _elementwise_op(other.data, out.grad, lambda x, y: x * y)
                self.grad = _add_gradients(self.grad, grad_contribution)
            if other.requires_grad and other.grad is not None:
                grad_contribution = _elementwise_op(self.data, out.grad, lambda x, y: x * y)
                other.grad = _add_gradients(other.grad, grad_contribution)
        
        out._backward = _backward
        return out
        
    def __truediv__(self, other : Union[Num, Tensor]) -> Parameter:
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False)
        
        # Check for division by zero
        if other.data == 0:
            raise ZeroDivisionError("Division by zero in tensor operation")
        data = _elementwise_op(self.data, other.data, lambda x, y: x / y)

        out = Parameter(data, requires_grad=self.requires_grad or other.requires_grad)
        out._prev = {self, other}

        def _backward():
            if self.requires_grad and self.grad is not None:
                grad_contribution = _elementwise_op(out.grad, other.data, lambda x, y: x / y)
                self.grad = _add_gradients(self.grad, grad_contribution)
            if other.requires_grad and other.grad is not None:
                grad_contribution = _elementwise_op(
                _elementwise_op(out.grad, self.data, lambda x, y: x * y),
                _elementwise_op(other.data, other.data, lambda x, y: x * y),
                lambda x, y: -x / y
            )
            other.grad = _add_gradients(other.grad, grad_contribution)
           
        out._backward = _backward
        return out

    def __pow__(self, other : Union[Num, Tensor]) -> Parameter:
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False)
        
        # Check for problematic cases
        if self.data == 0 and other.data < 0:
            raise ZeroDivisionError("Cannot raise 0 to a negative power")
        if self.data < 0 and not isinstance(other.data, int):
            raise ValueError("Cannot raise negative number to non-integer power")
        if self.data == 0 and other.data == 0:
            raise ValueError("0^0 is undefined")
        
        data = _elementwise_op(self.data, other.data, lambda x, y: x ** y)
        out = Parameter(data, requires_grad=self.requires_grad or other.requires_grad)
        out._prev = {self, other}

        def _backward():
            if self.requires_grad and self.grad is not None:
                grad_contribution = _elementwise_op(
                    _elementwise_op(out.grad, other.data, lambda x, y: x * y),
                    _elementwise_op(self.data, _elementwise_op(other.data, 1, lambda x, y: x - y), lambda x, y: x ** y),
                    lambda x, y: x * y
                )
                self.grad = _add_gradients(self.grad, grad_contribution)
            if other.requires_grad and other.grad is not None:
                def compute_other_grad(self_val, other_val, out_grad_val):
                    if self_val > 0:
                        return out_grad_val * math.log(self_val) * (self_val ** other_val)
                    else:
                        return 0.0
                
                grad_contribution = _elementwise_op(
                    _elementwise_op(self.data, other.data, lambda x, y: (x, y)),
                    out.grad,
                    lambda xy, grad: compute_other_grad(xy[0], xy[1], grad)
                )
                other.grad = _add_gradients(other.grad, grad_contribution)
        
        out._backward = _backward
        return out
        
    def __neg__(self) -> Parameter:
        out = Parameter(-self.data, requires_grad=self.requires_grad)
        out._prev = {self}

        def _backward():
            if self.requires_grad and self.grad is not None:
                neg_grad = _elementwise_op(out.grad, -1, lambda x, y: x * y)
                self.grad = _add_gradients(self.grad, neg_grad)
        
        out._backward = _backward
        return out
    
    def __radd__(self, other: Union[Num, Tensor]) -> Parameter:
        return self + other

    def __rsub__(self, other: Union[Num, Tensor]) -> Parameter:
        return Parameter(other, requires_grad=False) - self

    def __rmul__(self, other: Union[Num, Tensor]) -> Parameter:
        return self * other

    def __rtruediv__(self, other: Union[Num, Tensor]) -> Parameter:
        other_param = Parameter(other, requires_grad=False)
        return other_param / self

    def sum(self) -> Parameter:
        """Sum all elements in the parameter"""
        total = _sum_recursive(self.data)
        
        out = Parameter(total, requires_grad=self.requires_grad)
        out._prev = {self}
        
        def _backward():
            if self.requires_grad:
                grad_contribution = _ones_like(self.data)
                def scale_grad(grad_struct, scale):
                    if isinstance(grad_struct, list):
                        return [scale_grad(item, scale) for item in grad_struct]
                    else:
                        return grad_struct * scale
                
                scaled_grad = scale_grad(grad_contribution, out.grad)
                self.grad = _add_gradients(self.grad, scaled_grad)

        out._backward = _backward
        return out
        
    def __matmul__(self, other: Tensor) -> Parameter:
        """ Matrix multiplication with gradient tracking. """
        if not isinstance(other, Tensor):
            other = Tensor(other)
        
        # Forward pass is the same as the parent class
        result_data = super().__matmul__(other).data
        
        out = Parameter(result_data, requires_grad=self.requires_grad or other.requires_grad)
        out._prev = {self, other}

        def _backward():
            # Gradient for self: out.grad @ other.T
            if self.requires_grad:
                other_t = other.transpose()
                grad_self = Parameter(out.grad).__matmul__(other_t)
                self.grad = _add_gradients(self.grad, grad_self.data)

            # Gradient for other: self.T @ out.grad
            if other.requires_grad:
                self_t = self.transpose()
                grad_other = self_t.__matmul__(Parameter(out.grad))
                # Ensure other.grad is initialized if it's a non-Parameter tensor
                if other.grad is None:
                    other.grad = _zeros_like(other.data)
                other.grad = _add_gradients(other.grad, grad_other.data)
        
        out._backward = _backward
        return out
    
    @classmethod
    def zeros(cls, shape: Tuple[int, ...], requires_grad: bool = True) -> Parameter:
        """
        Create a parameter filled with zeros.
        Args: 
            shape (Tuple[int, ...]): Shape of the parameter to create
            requires_grad (bool): If True, the parameter will track gradients (default: True)
        Returns:
            Parameter: A parameter filled with zeros of the specified shape
        """
        def build_zeros(s):
            if len(s) == 1:
                return [0.0] * s[0]
            return [build_zeros(s[1:]) for _ in range(s[0])]
        
        if len(shape) == 0:
            data = 0.0
        else:
            data = build_zeros(shape)
        
        return cls(data, requires_grad=requires_grad)
    
    def detach(self) -> Tensor:
        """Detach returns a Tensor, not a Parameter"""
        return Tensor(self.data)
    
    def clone(self) -> Parameter:
        """Clone returns a Parameter"""
        return Parameter(self.data, requires_grad=self.requires_grad)
    
    # Override flatten to return Parameter
    def flatten(self) -> Parameter:
        flat_data = self._flatten_recursive(self.data)
        return Parameter(flat_data, requires_grad=self.requires_grad)
    
    def reshape(self, new_shape: Tuple[int, ...]) -> Parameter:
        current_elements = 1
        for dim in self.shape:
            current_elements *= dim

        new_elements = 1
        for dim in new_shape:
            new_elements *= dim
        
        if current_elements != new_elements:
            raise ValueError(f"Cannot reshape parameter of size {current_elements} to shape {new_shape} (size {new_elements})")
        
        flat_data = self._flatten_recursive(self.data)
        new_data = self._reshape_flat_to_nested(flat_data, new_shape)
        
        return Parameter(new_data, requires_grad=self.requires_grad)