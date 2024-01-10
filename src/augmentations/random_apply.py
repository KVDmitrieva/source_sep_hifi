import random
from typing import Callable

from torch import Tensor


class RandomApply:
    def __init__(self, augmentation: Callable, p: float):
        assert 0 <= p <= 1
        self.augmentation = augmentation
        self.p = p

    def __call__(self, data: Tensor) -> Tensor:
        if random.random() < self.p:
            return self.augmentation(data)
        else:
            return data
