from __future__ import annotations
from typing import List, Union, Callable, Optional, Set, Iterator, Tuple
import math

Num = Union[int, float]

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
        self.grad: Optional[Union[Num, List[Num]]] = _zeros_like(data)
        self._backward: Callable[[], None] = lambda: None
        self._prev: Set[Tensor] = set()

    def backward(self):
        """Compute gradients"""
        # if leaf has required_grad = False then no need to backpropagate for this whole graph
        if not self.requires_grad: 
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
        
        return Tensor(value)

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
        out._prev = {self, other}

        def _backward():
            self.grad = _add_gradients(self.grad, out.grad)
            other.grad = _add_gradients(other.grad, out.grad)
        
        out._backward = _backward
        return out
        
    def __sub__(self, other: Union[Num, Tensor]) -> Tensor:
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False)
        data = _elementwise_op(self.data, other.data, lambda x, y: x - y)

        out = Tensor(data, requires_grad=self.requires_grad or other.requires_grad)
        out._prev = {self, other}

        def _backward():
            self.grad = _add_gradients(self.grad, out.grad)
            neg_grad = _elementwise_op(out.grad, -1, lambda x, y: x * y)
            other.grad = _add_gradients(other.grad, neg_grad)
        
        out._backward = _backward
        return out
        
    def __mul__(self, other: Union[Num, Tensor]) -> Tensor:
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False)
        data = _elementwise_op(self.data, other.data, lambda x, y: x * y)
        
        out = Tensor(data, requires_grad=self.requires_grad or other.requires_grad)
        out._prev = {self, other}

        def _backward():
            grad_contribution = _elementwise_op(other.data, out.grad, lambda x, y: x * y)
            self.grad = _add_gradients(self.grad, grad_contribution)
            
            grad_contribution = _elementwise_op(self.data, out.grad, lambda x, y: x * y)
            other.grad = _add_gradients(other.grad, grad_contribution)
        
        out._backward = _backward
        return out
        
    def __truediv__(self, other : Union[Num, Tensor]) -> Tensor:
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False)
        
        # Check for division by zero
        if other.data == 0:
            raise ZeroDivisionError("Division by zero in tensor operation")
        data = _elementwise_op(self.data, other.data, lambda x, y: x / y)

        out = Tensor(data, requires_grad=self.requires_grad or other.requires_grad)
        out._prev = {self, other}

        def _backward():
            grad_contribution = _elementwise_op(out.grad, other.data, lambda x, y: x / y)
            self.grad = _add_gradients(self.grad, grad_contribution)
            
            grad_contribution = _elementwise_op(
                _elementwise_op(out.grad, self.data, lambda x, y: x * y),
                _elementwise_op(other.data, other.data, lambda x, y: x * y),
                lambda x, y: -x / y
            )
            other.grad = _add_gradients(other.grad, grad_contribution)
    
        out._backward = _backward
        return out

    def __pow__(self, other: Union[Num, Tensor]) -> Tensor:
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
        out._prev = {self, other}

        def _backward():
            grad_contribution = _elementwise_op(
                _elementwise_op(out.grad, other.data, lambda x, y: x * y),
                _elementwise_op(self.data, _elementwise_op(other.data, 1, lambda x, y: x - y), lambda x, y: x ** y),
                lambda x, y: x * y
            )
            self.grad = _add_gradients(self.grad, grad_contribution)
            
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
        
    def __neg__(self) -> Tensor:
        out = Tensor(-self.data, requires_grad=self.requires_grad)
        out._prev = {self}

        def _backward():
            neg_grad = _elementwise_op(out.grad, -1, lambda x, y: x * y)
            self.grad = _add_gradients(self.grad, neg_grad)
        
        out._backward = _backward
        return out
    
    def __radd__(self, other: Union[Num, Tensor]) -> Tensor:
        return self + other

    def __rsub__(self, other: Union[Num, Tensor]) -> Tensor:
        return Tensor(other) - self

    def __rmul__(self, other: Union[Num, Tensor]) -> Tensor:
        return self * other

    def __rtruediv__(self, other: Union[Num, Tensor]) -> Tensor:
        other_tensor = Tensor(other)
        return other_tensor / self

    # Bad code
    def __setitem__(self, index, value):
        """Value assignment"""
        if not isinstance(value, Tensor):
            value = Tensor(value)
        
        def _set_item_recursive(data, indices, val):
            if not indices:
                return val
            if isinstance(indices[0], slice):
                start, stop, step = indices[0].indices(len(data))
                if len(indices) == 1:
                    # Final level
                    if isinstance(val, list):
                        for i, idx in enumerate(range(start, stop, step)):
                            if i < len(val):
                                data[idx] = val[i]
                    else:
                        for idx in range(start, stop, step):
                            data[idx] = val
                else:
                    # Not final
                    for i, idx in enumerate(range(start, stop, step)):
                        if isinstance(val, list) and i < len(val):
                            _set_item_recursive(data[idx], indices[1:], val[i])
                        else:
                            _set_item_recursive(data[idx], indices[1:], val)
            else:
                if len(indices) == 1:
                    data[indices[0]] = val
                else:
                    _set_item_recursive(data[indices[0]], indices[1:], val)
        
        if isinstance(index, tuple):
            _set_item_recursive(self.data, list(index), value.data)
        elif isinstance(index, slice):
            start, stop, step = index.indices(len(self.data))
            if isinstance(value.data, list):
                for i, idx in enumerate(range(start, stop, step)):
                    if i < len(value.data):
                        self.data[idx] = value.data[i]
            else:
                for idx in range(start, stop, step):
                    self.data[idx] = value.data
        else:
            self.data[index] = value.data

    def __iadd__(self, other):
        """Support +="""
        if not isinstance(other, Tensor):
            other = Tensor(other)
        
        self.data = _elementwise_op(self.data, other.data, lambda x, y: x + y)
        
        old_backward = self._backward
        old_prev = self._prev.copy()
        
        def _backward():
            old_backward()
            other.grad = _add_gradients(other.grad, self.grad)
        
        self._backward = _backward
        self._prev = old_prev | {other}
        
        return self

    def sum(self) -> Tensor:
        """Sum all elements in the tensor"""
        total = _sum_recursive(self.data)
        
        out = Tensor(total, requires_grad=self.requires_grad)
        out._prev = {self}
        
        def _backward():
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

    def __matmul__(self, other: Tensor) -> Tensor:
        """ Matrix multiplication with gradient tracking. """
        if not isinstance(other, Tensor):
            other = Tensor(other)
        
        # Forward pass using parent class matmul
        result_data = super(Tensor, self).__matmul__(other).data
        
        out = Tensor(result_data, requires_grad=self.requires_grad or other.requires_grad)
        out._prev = {self, other}

        def _backward():
            def _transpose_last_two_dims(data):
                """Transpose last two dimensions"""
                if isinstance(data[0], list) and isinstance(data[0][0], list):
                    return [_transpose_last_two_dims(batch) for batch in data]
                else:
                    rows, cols = len(data), len(data[0])
                    return [[data[j][i] for j in range(rows)] for i in range(cols)]

            def _broadcast_matmul_raw(a_data, b_data):
                """Raw matrix multiplication with broadcasting"""
                temp_a = Tensor(a_data)
                temp_b = Tensor(b_data)
                result = temp_a.__matmul__(temp_b)
                return result.data
            
            def _reduce_broadcasted_dims(grad_data, target_shape, *other_shapes):
                """Reduce gradient to match target shape by summing over broadcasted dimensions"""
                def _sum_over_dim(data, dim):
                    """Sum over specified dimension"""
                    if dim == 0:
                        if not data:
                            return data
                        result = data[0]
                        for i in range(1, len(data)):
                            result = _elementwise_op(result, data[i], lambda x, y: x + y)
                        return result
                    else:
                        return [_sum_over_dim(sublist, dim - 1) for sublist in data]
                
                def get_shape(data):
                    if isinstance(data, list):
                        return [len(data)] + get_shape(data[0]) if data else []
                    return []
                
                grad_shape = get_shape(grad_data)
                
                if grad_shape == list(target_shape):
                    return grad_data
                
                padded_target = [1] * (len(grad_shape) - len(target_shape)) + list(target_shape)
                
                dims_to_sum = []
                for i, (grad_dim, target_dim) in enumerate(zip(grad_shape, padded_target)):
                    if target_dim == 1 and grad_dim > 1:
                        dims_to_sum.append(i)
                
                result = grad_data
                for dim in sorted(dims_to_sum, reverse=True):
                    result = _sum_over_dim(result, dim)
                
                while len(get_shape(result)) > len(target_shape):
                    if get_shape(result)[0] == 1:
                        result = result[0]
                    else:
                        break
                
                return result

            other_t = _transpose_last_two_dims(other.data)
            grad_self_full = _broadcast_matmul_raw(out.grad, other_t)
            grad_self = _reduce_broadcasted_dims(grad_self_full, self.shape, out.shape, other.shape)
            self.grad = _add_gradients(self.grad, grad_self)
                
            self_t = _transpose_last_two_dims(self.data)
            grad_other_full = _broadcast_matmul_raw(self_t, out.grad)
            grad_other = _reduce_broadcasted_dims(grad_other_full, other.shape, self.shape, out.shape)
            other.grad = _add_gradients(other.grad, grad_other)
        
        out._backward = _backward
        return out

    def detach(self) -> Tensor:
        return Tensor(self.data)

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

    @classmethod
    def ones(cls, shape: Tuple[int, ...], requires_grad: bool = False) -> Tensor:
        """
        Create a tensor filled with ones.
        Args: 
            shape (Tuple[int, ...]): Shape of the tensor to create
            requires_grad (bool): If True, the tensor will track gradients
        Returns:
            Tensor: A tensor filled with ones of the specified shape
        """
        def build_ones(s):
            if len(s) == 1:
                return [1.0] * s[0]
            return [build_ones(s[1:]) for _ in range(s[0])]
        
        if len(shape) == 0:
            data = 0.0
        else:
            data = build_ones(shape)
        
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

    def transpose_last_two_dims(self) -> Tensor:
        """Transpose the last two dimensions of a tensor (for batch matmul)"""

        def transpose_2d(matrix):
            """Transpose a 2D matrix"""
            if not matrix:
                return matrix
            if not isinstance(matrix[0], list):
                # Handle 1D case
                return [[row] for row in matrix]
            return [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]

        shape = self.shape
        if len(shape) < 2:
            raise ValueError("Need at least 2 dimensions to transpose")
        
        def transpose_recursive(data, dims_from_end):
            if dims_from_end == 2:
                if not isinstance(data, list) or not isinstance(data[0], list):
                    raise ValueError("Expected 2D structure at transpose level")
                return transpose_2d(data)
            else:
                # Recurse deeper
                return [transpose_recursive(item, dims_from_end - 1) for item in data]
        
        if len(shape) == 2:
            return self.transpose()
        
        else:
            transposed_data = transpose_recursive(self.data, len(shape))
            return Tensor(transposed_data, requires_grad=self.requires_grad)


class Parameter(Tensor):
    """New Param only requres_grad=True by default for now"""
    def __init__(self, data: Union[Num, List[Num]], requires_grad: bool = True):
        super().__init__(data, requires_grad=requires_grad)


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
    """Add gradients for gradient accumulation"""
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