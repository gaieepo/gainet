class Optimizer:
    def step(self, net):
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, lr=0.01):
        self.lr = lr

    def step(self, net):
        for param, grad in net.params_and_grads():
            param -= self.lr * grad
