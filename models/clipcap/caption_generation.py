from typing import List

import numpy as np
import torch
import torch.nn.functional as nnf
from tqdm import trange
from transformers import GPT2Tokenizer

from .clipcap import ClipCaptionModel
from .model_utils import remove_dots, remove_pad_token


def generate_beam(model,
                  tokenizer,
                  beam_size: int = 5,
                  prompt=None,
                  embed=None,
                  entry_length: int = 67,
                  temperature: float = 1.0,
                  stop_token: str = '.') -> List[str]:
    """
    Generates text using beam search with a given model and tokenizer.

    Args:
        model (torch.nn.Module): The language model used for text generation.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer for encoding/decoding text.
        beam_size (int, optional): The beam size for beam search. Defaults to 5.
        prompt (str, optional): The starting prompt for text generation. Defaults to None.
        embed (torch.Tensor, optional): The embedded prompt tensor for text generation. Defaults to None.
        entry_length (int, optional): The maximum number of words to generate. Defaults to 67.
        temperature (float, optional): The temperature value for sampling. Defaults to 1.0.
        stop_token (str, optional): The stop token used to terminate text generation. Defaults to '.'.

    Returns:
        List[str]: A list of generated text strings using beam search.
    """

    model.eval()
    stop_token_index = tokenizer.encode(stop_token)[0]
    tokens = None
    scores = None
    device = next(model.parameters()).device
    seq_lengths = torch.ones(beam_size, device=device)
    is_stopped = torch.zeros(beam_size, device=device, dtype=torch.bool)
    with torch.no_grad():
        if embed is not None:
            generated = embed
        else:
            if tokens is None:
                tokens = torch.tensor(tokenizer.encode(prompt))
                tokens = tokens.unsqueeze(0).to(device)
                generated = model.gpt.transformer.wte(tokens)
        for i in range(entry_length):
            outputs = model.gpt(inputs_embeds=generated)
            logits = outputs.logits
            logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
            logits = logits.softmax(-1).log()
            if scores is None:
                scores, next_tokens = logits.topk(beam_size, -1)
                generated = generated.expand(beam_size, *generated.shape[1:])
                next_tokens, scores = next_tokens.permute(1, 0), scores.squeeze(0)
                if tokens is None:
                    tokens = next_tokens
                else:
                    tokens = tokens.expand(beam_size, *tokens.shape[1:])
                    tokens = torch.cat((tokens, next_tokens), dim=1)
            else:
                logits[is_stopped] = -float(np.inf)
                logits[is_stopped, 0] = 0
                scores_sum = scores[:, None] + logits
                seq_lengths[~is_stopped] += 1
                scores_sum_average = scores_sum / seq_lengths[:, None]
                scores_sum_average, next_tokens = scores_sum_average.view(-1).topk(beam_size, -1)
                next_tokens_source = next_tokens // scores_sum.shape[1]
                seq_lengths = seq_lengths[next_tokens_source]
                next_tokens = next_tokens % scores_sum.shape[1]
                next_tokens = next_tokens.unsqueeze(1)
                tokens = tokens[next_tokens_source]
                tokens = torch.cat((tokens, next_tokens), dim=1)
                generated = generated[next_tokens_source]
                scores = scores_sum_average * seq_lengths
                is_stopped = is_stopped[next_tokens_source]
            next_token_embed = model.gpt.transformer.wte(next_tokens.squeeze()).view(generated.shape[0], 1, -1)
            generated = torch.cat((generated, next_token_embed), dim=1)
            is_stopped = is_stopped + next_tokens.eq(stop_token_index).squeeze()
            if is_stopped.all():
                break
    scores = scores / seq_lengths
    output_list = tokens.cpu().numpy()
    output_texts = [tokenizer.decode(output[:int(length)]) for output, length in zip(output_list, seq_lengths)]
    order = scores.argsort(descending=True)
    output_texts = [output_texts[i] for i in order]

    output_texts = remove_dots(output_texts)
    output_texts = remove_pad_token(output_texts, tokenizer.pad_token)

    return output_texts


def generate2(
        model,
        tokenizer,
        tokens=None,
        prompt=None,
        embed=None,
        entry_count: int = 1,
        entry_length: int = 67,  # maximum number of words
        top_p: float = 0.8,
        temperature: float = 1.0,
        stop_token: str = '.'
) -> str:
    """
    Generates text using top-p sampling with a given model and tokenizer.

    Args:
        model (torch.nn.Module): The language model used for text generation.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer for encoding/decoding text.
        tokens (torch.Tensor, optional): The starting token tensor for text generation. Defaults to None.
        prompt (str, optional): The starting prompt for text generation. Defaults to None.
        embed (torch.Tensor, optional): The embedded prompt tensor for text generation. Defaults to None.
        entry_count (int, optional): The number of text entries to generate. Defaults to 1.
        entry_length (int, optional): The maximum number of words to generate. Defaults to 67.
        top_p (float, optional): The top-p value for sampling. Defaults to 0.8.
        temperature (float, optional): The temperature value for sampling. Defaults to 1.0.
        stop_token (str, optional): The stop token used to terminate text generation. Defaults to '.'.

    Returns:
        str: The generated text string using top-p sampling.
    """
    model.eval()
    # generated_num = 0
    generated_list = []
    stop_token_index = tokenizer.encode(stop_token)[0]
    filter_value = -float("Inf")
    device = next(model.parameters()).device

    with torch.no_grad():

        for entry_idx in trange(entry_count):
            if embed is not None:
                generated = embed
            else:
                if tokens is None:
                    tokens = torch.tensor(tokenizer.encode(prompt))
                    tokens = tokens.unsqueeze(0).to(device)

                generated = model.gpt.transformer.wte(tokens)

            for i in range(entry_length):

                outputs = model.gpt(inputs_embeds=generated)
                logits = outputs.logits
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(nnf.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                                                    ..., :-1
                                                    ].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = filter_value
                next_token = torch.argmax(logits, -1).unsqueeze(0)
                next_token_embed = model.gpt.transformer.wte(next_token)
                if tokens is None:
                    tokens = next_token
                else:
                    tokens = torch.cat((tokens, next_token), dim=1)
                generated = torch.cat((generated, next_token_embed), dim=1)
                if stop_token_index == next_token.item():
                    break

            output_list = list(tokens.squeeze().cpu().numpy())
            output_text = tokenizer.decode(output_list)
            generated_list.append(output_text)

    generated_list = remove_dots(generated_list)
    generated_list = remove_pad_token(generated_list, tokenizer.pad_token)

    return generated_list[0]


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
            if use_beam_search:
                generated_texts.append(generate_beam(model, tokenizer, embed=prefix_embed)[0])
            else:
                generated_texts.append(generate2(model, tokenizer, embed=prefix_embed))

    return generated_texts
