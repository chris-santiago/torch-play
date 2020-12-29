import torch
import math

from get_mnist import load_data

MNIST = load_data().clean()


def log_softmax(x):
    return x - x.exp().sum(-1).log().unsqueeze(-1)


def model(batch, weights, bias):
    return log_softmax(torch.matmul(batch, weights) + bias)


x_train, y_train, x_valid, y_valid = map(torch.tensor, tuple(MNIST))

# We are initializing the weights here with Xavier initialisation (by multiplying with 1/sqrt(n)).
weights = torch.randn(784, 10) / math.sqrt(784)
weights.requires_grad_()
bias = torch.zeros(10, requires_grad=True)