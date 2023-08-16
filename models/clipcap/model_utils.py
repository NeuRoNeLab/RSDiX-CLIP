import re
import torch

from typing import List

from torch.nn import functional as f


def compute_loss(model, tokens, prefix, mask) -> torch.Tensor:
    outputs = model(tokens, prefix, mask)
    logits = outputs.logits[:, model.prefix_length - 1: -1]
    loss = f.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens.flatten(), ignore_index=0)

    return loss


def remove_dots(captions: List[str]) -> List[str]:
    # Remove extra "." if needed
    for i, text in enumerate(captions):
        if re.match(r".*\.{4,}", text):
            cleaned_txt = re.split(r"\.{4,}", text)[0]
            captions[i] = cleaned_txt + "."

    return captions


def remove_pad_token(captions: List[str], pad_token) -> List[str]:
    for i, caption in enumerate(captions):
        captions[i] = caption.replace(pad_token, "")

    return captions
