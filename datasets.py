from dataclasses import dataclass
from numpy import ndarray


@dataclass
class Mnist:
    """MNIST Dataset"""
    train_x: ndarray
    train_y: ndarray
    valid_x: ndarray
    valid_y: ndarray
