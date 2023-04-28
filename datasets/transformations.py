import numpy as np
import random
from torchvision.transforms import functional as F
import torch


class RandomSharpness(torch.nn.Module):
    """"""

    def __init__(self, mn: float = 0.5, mx: float = 1.5, p: float = 0.5):
        super().__init__()
        self.__mn = mn
        self.__mx = mx
        self.__p = p

    @property
    def max(self) -> float:
        return self.__mx

    @property
    def min(self) -> float:
        return self.__mn

    @property
    def p(self) -> float:
        return self.__p

    def __call__(self, sample):
        bit = np.random.binomial(n=1, p=self.__p)
        if bit == 1:
            return F.adjust_sharpness(sample, random.uniform(self.__mn, self.__mx))
        else:
            return sample
