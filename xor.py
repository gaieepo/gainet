import numpy as np

from gainet.train import train
from gainet.nn import Net
from gainet.layers import Linear, Tanh

inputs = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])

targets = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])

net = Net([Linear(xss=2, yss=2), Tanh(), Linear(xss=2, yss=2)])

train(net, inputs, targets)

for x, y in zip(inputs, targets):
    predicted = net.forward(x)
    print(x, predicted, y)
