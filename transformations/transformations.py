import random
from typing import Final

import numpy as np
import torch
from torchvision.transforms import functional as f
from transformers import MarianTokenizer, MarianMTModel, GPT2Tokenizer


def calculate_probability(n: int, p: float):
    """
    Calculate probability based on binomial distribution.

    Args:
        n (int): Number of trials.
        p (float): Probability of success.

    Returns:
        int: Randomly generated binomial value (0 or 1).
    """
    return np.random.binomial(n=n, p=p)


class RandomSharpness(torch.nn.Module):
    """
    Adjust image sharpness with a given probability.
    """

    def __init__(self, mn: float = 0.5, mx: float = 1.5, p: float = 0.5):
        """
        Initialize the RandomSharpness module.

        Args:
            mn (float): Lower bound limit of the sharpness range.
            mx (float): Upper bound limit of the sharpness range.
            p (float): Probability of adjusting sharpness.
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
        """
        Apply sharpness adjustment to an image with a certain probability.

        Args:
            sample: Input image.

        Returns:
            torch.Tensor: Transformed image.
        """
        bit = calculate_probability(n=1, p=self._p)

        return f.adjust_sharpness(sample, random.uniform(self._mn, self._mx)) if bit == 1 else sample


TGT_LANGS = ['fr', 'wa', 'frp', 'oc', 'ca', 'rm', 'lld', 'fur', 'lij', 'lmo', 'es', 'it', 'pt', 'gl', 'lad', 'an',
             'mwl', 'co', 'nap', 'scn', 'vec', 'sc', 'ro', 'la']
MAX_MODEL_LENGTH = 77
TOKENS_RANGE = 3


class BackTranslation:
    """
    Apply back translation to a text with a given probability.
    """

    def __init__(self, src_translator: str = "Helsinki-NLP/opus-mt-en-ROMANCE",
                 tgt_translator: str = "Helsinki-NLP/opus-mt-ROMANCE-en", p: float = 0.5):
        """
        Initialize the BackTranslation module.

        Args:
            src_translator (str): Name of the model for source language translation (back translation).
            tgt_translator (str): Name of the model for target language translation.
            p (float): Probability of applying back translation.
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
        """
        Translate a text using the specified translator.

        Args:
            sample (str): Input text.
            back (bool): If True, perform back translation.

        Returns:
            str: Translated text.
        """
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
        """
        Apply back translation to a text with a certain probability.

        Args:
            sample (str): Input text.

        Returns:
            str: Transformed text.
        """
        bit = calculate_probability(n=1, p=self._p)

        if bit == 1:
            # insert >>2 character language code<< at the beginning of the text to define the target language
            tgt_lang = random.choice(TGT_LANGS)
            sample = f">>{tgt_lang}<< " + sample
            return self._translate(self._translate(sample), back=True)
        else:
            return sample


PREFIX_LENGTH: Final[int] = 40


class GPT2Tokenization:
    """
        Tokenize and pad text sequences for GPT-2 models.
    """
    def __init__(self, prefix_length: int = PREFIX_LENGTH,
                 gpt2_type: str = "gpt2",
                 pad_token: str = None,
                 normalize_prefix: bool = False):
        """
        Initialize the GPT2Tokenization module.

        Args:
            prefix_length (int): Length of the prefix.
            gpt2_type (str): Type of GPT-2 model.
            pad_token (str): Padding token to use.
            normalize_prefix (bool): If True, normalize the prefix length.
        """
        self._tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type)
        self._prefix_length = prefix_length
        self._normalize_prefix = normalize_prefix

        if pad_token is not None:
            self._tokenizer.pad_token = pad_token
        elif self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

    def __call__(self, captions: str):
        """
        Tokenize and pad text sequences.

        Args:
            captions (str): Input text.

        Returns:
            tuple: Padded tokens and masks.
        """
        tokens = []
        max_seq_length = 0
        for c in captions:
            tokens.append(torch.tensor(self._tokenizer.encode(c), dtype=torch.int64))
            max_seq_length = max(tokens[-1].shape[0], max_seq_length)

        # padding
        masks = []
        padded_tokens = []
        for token in tokens:
            padded_token = token
            padding = max_seq_length - token.shape[0]
            if padding > 0:
                padded_token = torch.cat((token, torch.zeros(padding, dtype=torch.int64) - 1))
            elif padding < 0:
                padded_token = token[:max_seq_length]
            mask = padded_token.ge(0)  # mask is zero where we out of sequence
            padded_token[~mask] = 0
            mask = mask.float()
            mask = torch.cat((torch.ones(self._prefix_length), mask), dim=0)  # adding prefix mask
            padded_tokens.append(padded_token)
            masks.append(mask)

        # convert list to tensor
        padded_tokens = torch.stack(padded_tokens)
        masks = torch.stack(masks)

        return padded_tokens, masks
