from .data import BatchIterator
from .optim import SGD
from .loss import MSE


def train(
        net,
        xs,
        ys,
        num_epochs=5000,
        iterator=BatchIterator(),
        loss=MSE(),
        optimizer=SGD(),
):
    for epoch in range(num_epochs):
        epoch_loss = 0.0

        for batch in iterator(xs, ys):
            predicted = net.forward(batch.xs)
            epoch_loss += loss.loss(predicted, batch.ys)
            grad = loss.grad(predicted, batch.ys)
            net.backward(grad)
            optimizer.step(net)
        print(epoch, epoch_loss)
