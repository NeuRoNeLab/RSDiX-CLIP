import random

import translators as ts
import numpy as np
import torch

from torchvision.transforms import functional as F


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


class BackTranslation:
    """"""

    def __init__(self, from_language: str, to_language: str, translator: str = "google", p: float = 0.5):
        self.__from_language = from_language
        self.__to_language = to_language
        self.__translator = translator
        self.__p = p

    @property
    def from_language(self) -> str:
        return self.__from_language

    @property
    def to_language(self) -> str:
        return self.__to_language

    @property
    def translator(self) -> str:
        return self.__translator

    @property
    def p(self) -> float:
        return self.__p

    def _translate(self, sample, from_language=None, to_language=None):
        if from_language is None:
            from_language = self.__from_language
        if to_language is None:
            to_language = self.__to_language

        return ts.translate_text(sample, translator=self.__translator,
                                 from_language=from_language, to_language=to_language)

    def __call__(self, sample) -> str:
        bit = np.random.binomial(n=1, p=self.__p)
        if bit == 1:
            return self._translate(self._translate(sample), from_language=self.__to_language,
                                   to_language=self.from_language)
        else:
            return sample
