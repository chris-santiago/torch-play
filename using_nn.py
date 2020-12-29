import math

import torch
import torch.nn as nn

from get_mnist import load_data

MNIST = load_data().clean(binarize=False)


class MnistLogistic(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(784, 10) / math.sqrt(784))
        self.bias = nn.Parameter(torch.zeros(10))
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, xb):
        return xb.double() @ self.weights.double() + self.bias


class MnistTrainer:
    def __init__(self, model, epochs, lr):
        self.model = model
        self.epochs = epochs
        self.lr = lr

    def fit(self, x_train, y_train, bs):
        for epoch in range(self.epochs):
            for i in range((x_train.shape[0] - 1) // bs + 1):
                start_i = i * bs
                end_i = start_i + bs
                xb = x_train[start_i:end_i]
                yb = y_train[start_i:end_i]
                pred = self.model.forward(xb)
                loss = self.model.loss_func(pred, yb)
                loss.backward()
                with torch.no_grad():
                    for param in self.model.parameters():
                        param -= param.grad * self.lr
                    self.model.zero_grad()

    def predict(self, x_valid):
        return self.model.forward(x_valid)


def accuracy(preds, actual):
    return (preds.argmax(1) == actual).float().mean()


def main():
    x_train, y_train, x_valid, y_valid = map(torch.tensor, tuple(MNIST))
    model = MnistLogistic()
    learner = MnistTrainer(model, epochs=20, lr=0.1)
    learner.fit(x_train, y_train.long(), 256)
    fitted = learner.predict(x_train)
    fitted_loss = model.loss_func(fitted, y_train.long())
    fitted_acc = accuracy(fitted, y_train.long())
    print(f'Training Loss: {fitted_loss}.  Training Accuracy: {fitted_acc}.')
    pred = learner.predict(x_valid)
    loss = model.loss_func(pred, y_valid.long())
    acc = accuracy(pred, y_valid.long())
    print(f'Validation Loss: {loss}.  Validation Accuracy: {acc}.')


if __name__ == '__main__':
    main()
