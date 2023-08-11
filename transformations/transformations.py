import random
from typing import Final

import numpy as np
import torch
from torchvision.transforms import functional as f
from transformers import MarianTokenizer, MarianMTModel, GPT2Tokenizer


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
MAX_MODEL_LENGTH = 77
TOKENS_RANGE = 3


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
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._p = p
        self._src_translator = MarianMTModel.from_pretrained(src_translator).to(self._device)
        self._src_tokenizer = MarianTokenizer.from_pretrained(src_translator)
        self._tgt_translator = MarianMTModel.from_pretrained(tgt_translator).to(self._device)
        self._tgt_tokenizer = MarianTokenizer.from_pretrained(tgt_translator)

    @property
    def p(self) -> float:
        return self._p

    def _translate(self, sample, back: bool = False):
        tokens = self._src_tokenizer(sample, return_tensors="pt", padding=True) if back is not True else \
            self._tgt_tokenizer(sample, return_tensors="pt", padding=True)
        tokens = tokens.to(self._device)

        translated = self._src_translator.generate(**tokens, max_new_tokens=MAX_MODEL_LENGTH) if back is not True else \
            self._tgt_translator.generate(**tokens, max_new_tokens=MAX_MODEL_LENGTH)

        # if the translated sample contains more tokens than the specified threshold, return the original
        # sample
        translated = translated if len(translated) <= (len(tokens) * TOKENS_RANGE) else tokens
        translated = translated.to("cpu")

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


PREFIX_LENGTH: Final[int] = 40
GPT_MAX_LENGTH = 100


class GPT2Tokenization:

    def __init__(self, prefix_length: int = PREFIX_LENGTH,
                 gpt2_type: str = "gpt2",
                 pad_token: str = None,
                 normalize_prefix: bool = False):
        self._tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type)
        self._prefix_length = prefix_length
        self._normalize_prefix = normalize_prefix

        if pad_token is not None:
            self._tokenizer.pad_token = pad_token
        elif self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

    def __call__(self, captions: str):
        caption_tokens = []
        for c in captions:
            caption_tokens.append(torch.tensor(self._tokenizer.encode(c, padding='max_length',
                                                                      max_length=GPT_MAX_LENGTH), dtype=torch.int64))

        caption_tokens = torch.stack(caption_tokens)
        mask = caption_tokens.ge(0)  # mask is zero where we out of sequence
        caption_tokens[~mask] = 0
        mask = mask.float()
        # adding prefix mask
        mask = torch.cat((torch.ones(caption_tokens.shape[0], self._prefix_length), mask), dim=1)

        return caption_tokens, mask
