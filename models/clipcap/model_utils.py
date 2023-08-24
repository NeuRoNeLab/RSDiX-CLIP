import re
import torch

from typing import List

from torch.nn import functional as f


def compute_loss(model, tokens, prefix, mask) -> torch.Tensor:
    """
    Compute the loss for a sequence generation task using the given model.

    Args:
        model (clipcap.ClipCaptioningModel): The sequence generation model.
        tokens (torch.Tensor): The input tokens for the model.
        prefix (torch.Tensor): The prefix tokens for the sequence.
        mask (torch.Tensor): The attention mask for the input tokens.

    Returns:
        torch.Tensor: The computed loss.
    """
    outputs = model(tokens, prefix, mask)
    logits = outputs.logits[:, model.prefix_length - 1: -1]
    loss = f.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens.flatten(), ignore_index=0)

    return loss


def remove_dots(captions: List[str]) -> List[str]:
    """
    Remove extra periods from a list of captions.

    Args:
        captions (List[str]): A list of caption strings.

    Returns:
        List[str]: The list of captions with extra periods removed.

    Example:
        captions = ["Hello....", "This is a test....."]
        cleaned_captions = remove_dots(captions)
        print(cleaned_captions)
        ["Hello.", "This is a test."]
    """
    for i, text in enumerate(captions):
        if re.match(r".*\.{4,}", text):
            cleaned_txt = re.split(r"\.{4,}", text)[0]
            captions[i] = cleaned_txt + "."

    return captions


def remove_pad_token(captions: List[str], pad_token) -> List[str]:
    """
    Remove a specified pad token from a list of captions.

    Args:
        captions (List[str]): A list of caption strings.
        pad_token (str): The pad token to remove from captions.

    Returns:
        List[str]: The list of captions with the specified pad token removed.

    Example:
        captions = ["This is an [PAD] example.", "[PAD] Padding test [PAD]"]
        cleaned_captions = remove_pad_token(captions, "[PAD]")
        print(cleaned_captions)
        ["This is an example.", " Padding test "]
    """
    for i, caption in enumerate(captions):
        captions[i] = caption.replace(pad_token, "")

    return captions
