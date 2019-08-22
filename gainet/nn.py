class Net:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, xs):
        for layer in self.layers:
            xs = layer.forward(xs)

        return xs

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

        return grad

    def params_and_grads(self):
        for layer in self.layers:
            for name, param in layer.params.items():
                grad = layer.grads[name]
                yield param, grad
