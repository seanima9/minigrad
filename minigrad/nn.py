import numpy as np
from minigrad.tensor import Tensor
from abc import ABC, abstractmethod


class Layer:
    def __init__(self, in_size, out_size, activation='relu', weight_init='he', precision=np.float32,):
        initializers = {
            'std_normal': lambda shape: np.random.normal(0, 1, shape),
            'he': lambda shape: np.random.randn(*shape) * np.sqrt(2.0 / shape[1]),
            'xavier': lambda shape: np.random.randn(*shape) * np.sqrt(2.0 / (shape[1] + shape[0])),
        }
        if weight_init not in initializers:
            raise ValueError(f"Unknown weight initialization method: {weight_init}")
        weight_init_fn = initializers[weight_init]

        self.weights = Tensor(weight_init_fn((in_size, out_size)), requires_grad=True, precision=precision)
        self.bias = Tensor(np.zeros((1, out_size), dtype=precision), requires_grad=True, precision=precision)
        self.activation = activation
        self.weight_init = weight_init
        self.precision = precision
        
    def __call__(self, x):
        z = x @ self.weights + self.bias
        activation_fn = getattr(z, self.activation)
        if activation_fn is None:
            raise ValueError(f"Unknown activation function: {self.activation}")
        out = activation_fn()
        return out
        
    def parameters(self):
        return [self.weights, self.bias]

    def __repr__(self):
        return f"Layer(in_size={self.weights.shape[1]}, out_size={self.weights.shape[0]}, activation={self.activation}, weight_init={self.weight_init}, precision={self.precision})"


class MLP:
    def __init__(self, layers, precision=np.float32, weight_init='he', activation='relu'):
        self.layers = []
        for i in range(len(layers) - 1):
            self.layers.append(Layer(layers[i], layers[i + 1], activation=activation,
                                     weight_init=weight_init, precision=precision)) 
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def __repr__(self):
        return f"MLP(layers={self.layers})"

    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params


class Optimizer(ABC):
    def __init__(self, parameters):
        self.parameters = parameters
    
    def zero_grad(self):
        for p in self.parameters:
            p.grad = np.zeros_like(p.data)
    
    @abstractmethod
    def step(self):
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, parameters, lr=0.01):
        super().__init__(parameters)
        self.lr = lr

    def step(self):
        for p in self.parameters:
            p.data -= self.lr * p.grad

    def __repr__(self):
        return f"SGD(lr={self.lr})"


def mse_loss(pred, target): return ((pred - target) ** 2).mean()

