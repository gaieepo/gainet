import numpy as np

from gainet.train import train
from gainet.nn import Net
from gainet.layers import Linear, Tanh
from gainet.optim import SGD


def fizz_buzz_encode(x):
    if x % 15 == 0:
        return [0, 0, 0, 1]
    elif x % 5 == 0:
        return [0, 0, 1, 0]
    elif x % 3 == 0:
        return [0, 1, 0, 0]
    else:
        return [1, 0, 0, 0]


def binary_encode(x):
    return [x >> i & 1 for i in range(10)]


inputs = np.array([binary_encode(x) for x in range(101, 1024)])

targets = np.array([fizz_buzz_encode(x) for x in range(101, 1024)])

net = Net([Linear(xss=10, yss=50), Tanh(), Linear(xss=50, yss=4)])

train(net, inputs, targets, num_epochs=500, optimizer=SGD(lr=0.001))

for x in range(1, 101):
    predicted = net.forward(binary_encode(x))
    predicted_idx = np.argmax(predicted)
    actual_idx = np.argmax(fizz_buzz_encode(x))
    labels = [str(x), 'fizz', 'buzz', 'fizzbuzz']
    print(x, labels[predicted_idx], labels[actual_idx])
