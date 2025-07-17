import numpy as np
from collections import deque

class Tensor:
    def __init__(self, data, _parents=(), requires_grad=False, precision=np.float32):
        self.data = np.array(data, dtype=precision)
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(self.data, dtype=precision)
        self.shape = self.data.shape
        self._backward_fn = lambda: None
        self.parents = set(_parents)

    def __repr__(self):
        return f"Tensor(shape={self.shape}, requires_grad={self.requires_grad})"

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        assert self.shape == other.shape, f"Shape mismatch for addition: {self.shape} vs {other.shape}"
        out = Tensor(self.data + other.data , _parents=(self,other), requires_grad=self.requires_grad) 
        
        def _backward_fn():
            if self.requires_grad:
                self.grad += out.grad # += to allow for nodes with multiple parents
            if other.requires_grad:
                other.grad += out.grad
        out._backward_fn = _backward_fn

        return out

    def __mul__(self, other): # element-wise mult
        other = other if isinstance(other, Tensor) else Tensor(other)
        assert self.shape == other.shape, f"Shape mismatch for hadamard product: {self.shape} vs {other.shape}"
        
        out = Tensor(self.data * other.data, _parents=(self, other), requires_grad=self.requires_grad)

        def _backward_fn():
            if self.requires_grad:
                self.grad += out.grad * other.data
            if other.requires_grad:
                other.grad += out.grad * self.data
        out._backward_fn = _backward_fn
        
        return out

    def __pow__(self, power):
        assert isinstance(power, (int, float)), "Power must be a scalar"
        out = Tensor(self.data ** power, _parents=(self,), requires_grad=self.requires_grad)

        def _backward_fn():
            if self.requires_grad:
                self.grad += out.grad * (power * self.data ** (power - 1))
        out._backward_fn = _backward_fn

        return out
    
    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        assert self.shape[-1] == other.shape[0], f"Shape mismatch for matrix multiplication: {self.shape} vs {other.shape}"

        out = Tensor(np.matmul(self.data, other.data), _parents=(self, other), requires_grad=self.requires_grad)
        def _backward_fn():
            if self.requires_grad:
                self.grad += out.grad @ other.data.T
            if other.requires_grad:
                other.grad += self.data.T @ out.grad
        out._backward_fn = _backward_fn

        return out

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self, other):
        return self * (other ** -1)
   
    def __radd__(self, other):
        return self + other

    def __rsub__(self, other):
        return other + (-self)
    
    def __rmul__(self, other):
        return self * other

    def __rtruediv__(self, other):
        return other * (self**-1)
    
    def relu(self):
        out = Tensor(np.maximum(0, self.data), _parents=(self,), requires_grad=self.requires_grad)

        def _backward_fn():
            if self.requires_grad:
                self.grad += out.grad * (self.data > 0).astype(self.data.dtype)
        out._backward_fn = _backward_fn

        return out

    def sigmoid(self):
        sig = 1 / (1 + np.exp(-self.data))
        sig_deriv = sig * (1 - sig)
        out = Tensor(sig, _parents=(self,), requires_grad=self.requires_grad)

        def _backward_fn():
            if self.requires_grad:
                self.grad += out.grad * sig_deriv
        out._backward_fn = _backward_fn

        return out

    def backward(self):
        reverse_topo = []
        visited = set()
        queue = deque([self])
        while queue:
            node = queue.popleft()
            if node in visited:
                continue
            visited.add(node)
            reverse_topo.append(node) # adds to right end
            for parent in node.parents:
                if parent not in visited:
                    queue.append(parent)
        
        if self.requires_grad: self.grad = np.ones_like(self.data, dtype=self.data.dtype)
        for node in reverse_topo:
            if node.requires_grad:
                node._backward_fn()
         
