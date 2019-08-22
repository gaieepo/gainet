import numpy as np


class Layer:
    def __init__(self):
        self.params = {}
        self.grads = {}

    def forward(self, xs):
        raise NotImplementedError

    def backward(self, grad):
        raise NotImplementedError


class Linear(Layer):
    def __init__(self, xss, yss):
        super().__init__()
        self.params['w'] = np.random.randn(xss, yss)
        self.params['b'] = np.random.randn(yss)

    def forward(self, xs):
        self.xs = xs

        return xs @ self.params['w'] + self.params['b']

    def backward(self, grad):
        self.grads['b'] = np.sum(grad, axis=0)
        self.grads['w'] = self.xs.T @ grad

        return grad @ self.params['w'].T


class Activation(Layer):
    def __init__(self, f, f_prime):
        super().__init__()
        self.f = f
        self.f_prime = f_prime

    def forward(self, xs):
        self.xs = xs

        return self.f(xs)

    def backward(self, grad):
        return self.f_prime(self.xs) * grad


def tanh(x):
    return np.tanh(x)


def tanh_prime(x):
    y = tanh(x)

    return 1 - y**2


class Tanh(Activation):
    def __init__(self):
        super().__init__(tanh, tanh_prime)
