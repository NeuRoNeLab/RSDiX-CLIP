import torch
import random

import numpy as np

from torchvision.transforms import functional as f
from transformers import MarianTokenizer, MarianMTModel, GPT2Tokenizer

from utils import PREFIX_LENGTH


def calculate_probability(n: int, p: float):
    return np.random.binomial(n=n, p=p)


class RandomSharpness(torch.nn.Module):
    """ This class adjusts an image sharpness with a given probability. """

    def __init__(self, mn: float = 0.5, mx: float = 1.5, p: float = 0.5):
        """
          Args:
               mn (float): the lower bound limit of the range in which the sharpness value is uniformly chosen.
               mx (float): the upper bound limit of the range in which the sharpness value is uniformly chosen
               p (float): the probability with which to adjust the sharpness.
          """
        super().__init__()
        self._mn = mn
        self._mx = mx
        self._p = p

    @property
    def max(self) -> float:
        return self.mx

    @property
    def min(self) -> float:
        return self.__mn

    @property
    def p(self) -> float:
        return self.__p

    def __call__(self, sample):
        bit = calculate_probability(n=1, p=self._p)

        return f.adjust_sharpness(sample, random.uniform(self._mn, self._mx)) if bit == 1 else sample


TGT_LANGS = ['fr', 'wa', 'frp', 'oc', 'ca', 'rm', 'lld', 'fur', 'lij', 'lmo', 'es', 'it', 'pt', 'gl', 'lad', 'an',
             'mwl', 'co', 'nap', 'scn', 'vec', 'sc', 'ro', 'la']


class BackTranslation:
    """ This class applies back translation on a text with a given probability. """

    def __init__(self, src_translator: str = "Helsinki-NLP/opus-mt-en-ROMANCE",
                 tgt_translator: str = "Helsinki-NLP/opus-mt-ROMANCE-en", p: float = 0.5):
        """
            Args:
               src_translator (str): the name of the model that will be used to translate to the source language
               (back translation).
               tgt_translator (str): the name of the target that will be used to translate to the target language
               (translation).
                p (float):  the probability with which to apply back translation.
        """
        self._p = p
        self._src_translator = MarianMTModel.from_pretrained(src_translator)
        self._src_tokenizer = MarianTokenizer.from_pretrained(src_translator)
        self._tgt_translator = MarianMTModel.from_pretrained(tgt_translator)
        self._tgt_tokenizer = MarianTokenizer.from_pretrained(tgt_translator)

    @property
    def p(self) -> float:
        return self._p

    def _translate(self, sample, back: bool = False):
        translated = self._src_translator.generate(**self._src_tokenizer(sample, return_tensors="pt", padding=True),
                                                   max_new_tokens=512) \
            if back is not True else \
            self._tgt_translator.generate(**self._tgt_tokenizer(sample, return_tensors="pt", padding=True),
                                          max_new_tokens=512)

        # translated text
        return [self._src_tokenizer.decode(t, skip_special_tokens=True) for t in translated][0] \
            if back is not True else \
            [self._tgt_tokenizer.decode(t, skip_special_tokens=True) for t in translated][0]

    def __call__(self, sample: str):
        bit = calculate_probability(n=1, p=self._p)

        if bit == 1:
            # insert >>2 character language code<< at the beginning of the text to define the target language
            tgt_lang = random.choice(TGT_LANGS)
            sample = f">>{tgt_lang}<< " + sample
            return self._translate(self._translate(sample), back=True)
        else:
            return sample


class GPT2Tokenizer:

    def __init__(self, prefix_length: int = PREFIX_LENGTH,
                 gpt2_type: str = "gpt2",
                 normalize_prefix: bool = False):
        self._tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type)
        self._prefix_length = prefix_length
        self._normalize_prefix = normalize_prefix
        self._caption_tokens = []
        self._max_seq_len = 0

    def _pad_tokens(self):
        pass

    def __call__(self, captions: str):
        max_seq_len = 0
        all_len = []
        for c in captions:
            self._caption_tokens.append([torch.tensor(self._tokenizer.encode(c), dtype=torch.int64)])
            seq_lenghts = [ct.shape[0] for ct in self._caption_tokens[-1]]
            all_len.extend(seq_lenghts)
            max_seq_len = max(max_seq_len, *seq_lenghts)
            max_seq_len = max(max_seq_len, *seq_lenghts)

        all_len = torch.tensor(all_len).float()
        self.max_seq_len = min(int(all_len.mean() + all_len.std() * 10), max_seq_len)


        

