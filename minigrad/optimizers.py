import numpy as np
from abc import ABC, abstractmethod


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


class Adam(Optimizer):
    pass
