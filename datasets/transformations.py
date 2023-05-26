import random
import time

import translators as ts
import numpy as np
import torch

from torchvision.transforms import functional as F
from requests import exceptions as re


class RandomSharpness(torch.nn.Module):
    """ This class adjusts an image sharpness with a given probability. """

    def __init__(self, mn: float = 0.5, mx: float = 1.5, p: float = 0.5):
        """
            Arguments:
                 mn (float): the lower bound limit of the range in which the sharpness value is uniformly chosen.
                 mx (float): the upper bound limit of the range in which the sharpness value is uniformly chosen
                 p (float): the probability with which to adjust the sharpness.
        """
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
    """ This class applies back translation on a text with a given probability. """
    __api_timestamps = {}
    BACK_TRANSLATION_TRANSLATORS = ["google", "bing"]
    BACK_TRANSLATION_LANGUAGES = ["zh", "ar", "ru", "de", "it", "fr", "ar", "es", "ja"]

    def __init__(self, from_language: str, p: float = 0.5, timeout: float = 1.0):
        """
            Arguments:
                 from_language (str): the language to translate from.
                 p (float): the probability with which to apply back translation.
                 timeout (float): the period of time between one API call and another (in seconds).
        """
        self.__from_language = from_language
        self.__p = p
        self.__timeout = timeout

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

    @property
    def timeout(self) -> float:
        return self.__timeout

    def _translate(self, sample, from_language=None, to_language=None) -> str:
        if from_language is None:
            from_language = self.__from_language
        if to_language is None:
            to_language = self.__to_language

        return ts.translate_text(sample, translator=self.__translator,
                                 from_language=from_language, to_language=to_language)

    def _is_api_callable(self) -> bool:
        current_time = time.time()
        last_api_call = self.__api_timestamps.get(self.__translator)

        if last_api_call is None or (current_time - last_api_call) > self.__timeout:
            self.__api_timestamps[self.__translator] = current_time
            return True
        else:
            # search for a callable API, if none is found return False
            for key, trs in self.__api_timestamps.items():
                if key == self.__translator:
                    continue

                if (current_time - trs) > self.__timeout:
                    self.__translator = key
                    self.__api_timestamps[self.__translator] = current_time
                    return True

        return False

    def __call__(self, sample) -> str:
        self.__translator = self.BACK_TRANSLATION_TRANSLATORS[
            random.randint(0, len(self.BACK_TRANSLATION_TRANSLATORS) - 1)]
        self.__to_language = self.BACK_TRANSLATION_LANGUAGES[
            random.randint(0, len(self.BACK_TRANSLATION_LANGUAGES) - 1)]
        bit = np.random.binomial(n=1, p=self.__p)
        # check if API call's timeout has passed, if so translate otherwise choose another API and repeat.
        # if no API call's timeout has passed, return the normal sample.
        if bit == 1 and self._is_api_callable():
            try:
                return self._translate(self._translate(sample), from_language=self.__to_language,
                                       to_language=self.from_language)
            except (TypeError, IndexError, re.JSONDecodeError, re.HTTPError, re.ConnectionError):
                return sample
        else:
            return sample
