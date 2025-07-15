import numpy as np

class Tensor:
    def __init__(self, data, _children=(), requires_grad=False, precision=np.float32):
        self.data = np.array(data, dtype=precision)
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(self.data, dtype=precision)
        self.shape = self.data.shape
        self._backward_fn = lambda: None
        _ = _children

    def __repr__(self):
        return f"Tensor(shape={self.shape}, requires_grad={self.requires_grad})"

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        assert self.shape == other.shape, f"Shape mismatch for addition: {self.shape} vs {other.shape}"
        out = Tensor(self.data + other.data , _children=(self,other), requires_grad=self.requires_grad) 
        
        def _backward_fn():
            if self.requires_grad:
                self.grad += out.grad
            if other.requires_grad:
                other.grad += out.grad
        out._backward_fn = _backward_fn

        return out

    def __mul__(self, other): # hadamard product
        other = other if isinstance(other, Tensor) else Tensor(other)
        assert self.shape == other.shape, f"Shape mismatch for hadamard product: {self.shape} vs {other.shape}"
        
        out = Tensor(self.data * other.data, _children=(self, other), requires_grad=self.requires_grad)

        def _backward_fn():
            if self.requires_grad:
                self.grad += out.grad * other.data
            if other.requires_grad:
                other.grad += out.grad * self.data
        out._backward_fn = _backward_fn
        
        return out
    def __pow__(self, power):
        assert isinstance(power, (int, float)), "Power must be a scalar"
        out = Tensor(self.data ** power, _children=(self,), requires_grad=self.requires_grad)

        def _backward_fn():
            if self.requires_grad:
                self.grad += out.grad * (power * self.data ** (power - 1))
        out._backward_fn = _backward_fn

        return out
    
    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        assert self.shape[-1] == other.shape[0], f"Shape mismatch for matrix multiplication: {self.shape} vs {other.shape}"

        out = Tensor(np.matmul(self.data, other.data), _children=(self, other), requires_grad=self.requires_grad)
        def _backward_fn():
            pass

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self, other):
        return self * (other ** -1)

