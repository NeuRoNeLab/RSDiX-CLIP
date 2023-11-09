from typing import List

import torch
from transformers import GPT2Tokenizer

from .clipcap import ClipCaptionModel


def generate_caption(imgs,
                     model: ClipCaptionModel,
                     tokenizer: GPT2Tokenizer,
                     clip_encoder,
                     use_beam_search: bool = True) -> List[str]:
    """
    Generates a caption for a given image using a pre-trained CLIPCap model.

    Args:
        imgs (torch.Tensor): The input images for caption generation.
        model (ClipCaptionModel): The image-captioning model for generating captions.
        tokenizer (GPT2Tokenizer): The tokenizer for encoding/decoding text.
        clip_encoder (torch.nn.Module): The CLIP model used for encoding images.
        use_beam_search (bool, optional): Whether to use beam search for text generation. Defaults to True.

    Returns:
        List[str]: The generated captions for the input images.
    """

    generated_texts = []
    with torch.no_grad():
        imgs = imgs if len(imgs.shape) > 3 else imgs.unsqueeze(0)
        clip_prefix = clip_encoder.encode_image(imgs)
        # normalize
        clip_prefix /= clip_prefix.norm(p=2, dim=-1, keepdim=True)
        for idx in range(clip_prefix.shape[0]):
            prefix_embed = model.clip_project(clip_prefix[idx]).reshape(1, 40, -1)
            outputs = model.gpt.generate(**prefix_embed)
            generated_texts.append(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
    return generated_texts
