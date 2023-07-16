import torch
import math

from typing import final
from sentence_transformers.util import cos_sim

REDUCTIONS: final = frozenset(["mean", "average", "avg", "sum", "add", "none"])


@torch.no_grad()
# Source: https://github.com/facebookresearch/swav/blob/5e073db0cc69dea22aa75e92bfdd75011e888f28/main_swav.py#L354
def sinkhorn(self, out):
    q = torch.exp(out / 0.05).t()  # q is k-by-b for consistency with notations from our paper
    b = q.shape[1]  # number of samples to assign
    k = q.shape[0]  # how many prototypes

    # make the matrix sums to 1
    sum_q = torch.sum(q)
    q /= sum_q

    for it in range(3):
        # normalize each row: total weight per prototype must be 1/k
        sum_of_rows = torch.sum(q, dim=1, keepdim=True)
        q /= sum_of_rows
        q /= k

        # normalize each column: total weight per sample must be 1/b
        q /= torch.sum(q, dim=0, keepdim=True)
        q /= b

    q *= b  # the columns must sum to 1 so that q is an assignment
    return q.t()


def compute_similarities(i_emb, t_emb):
    sim_ii, sim_tt = i_emb @ i_emb.t(), t_emb @ t_emb.t()
    sim_it, sim_ti = i_emb @ t_emb.t(), t_emb @ t_emb.t()
    return sim_ii, sim_tt, sim_it, sim_ti


def compute_st_similarities(clip_image_embeddings, clip_text_embeddings, st_embeddings):
    image_image_similarities = cos_sim(clip_image_embeddings, clip_image_embeddings)
    image_text_similarities = cos_sim(clip_image_embeddings, clip_text_embeddings)
    text_text_similarities_clip = cos_sim(clip_text_embeddings, clip_text_embeddings)
    text_text_similarities_st = cos_sim(st_embeddings, st_embeddings)

    return image_image_similarities, image_text_similarities, text_text_similarities_clip, text_text_similarities_st


def compute_mse_similarities(image_image_similarities: torch.Tensor,
                             image_text_similarities: torch.Tensor,
                             text_text_similarities_clip: torch.Tensor,
                             text_text_similarities_st: torch.Tensor,
                             reduction: str = "mean") -> torch.Tensor:
    if reduction not in REDUCTIONS:
        raise ValueError(f"'reduction' parameter must be one of {REDUCTIONS}. {reduction} given.")

    if reduction == "average":
        reduction = "mean"

    mse_loss = torch.nn.MSELoss()
    ii_mse = mse_loss(image_image_similarities, text_text_similarities_st)
    it_mse = mse_loss(image_text_similarities, text_text_similarities_st)
    tt_mse = mse_loss(text_text_similarities_clip, text_text_similarities_st)
    mse_tensor = torch.stack([ii_mse, it_mse, tt_mse])

    if reduction == "mean" or reduction == "average" or reduction == "avg":
        return torch.mean(mse_tensor)
    if reduction == "sum" or reduction == "add":
        return torch.mean(mse_tensor)
    else:
        return mse_tensor


def get_image_caption_chunks(image, caption, batch_size):
    n = math.ceil(len(image) // batch_size)
    image_chunks = torch.chunk(image, n)
    caption_chunks_ids = torch.chunk(torch.arange(len(image)), n)

    # getting the caption for the current gpu
    # then stacking them to match image_chunks' type
    caption_chunks = []
    caption = list(caption)
    for s in caption_chunks_ids[0]:
        caption_chunks.append(caption[s.item()])
    caption_chunks = (torch.stack(caption_chunks),)

    return image_chunks, caption_chunks


def ema(s, t):
    return s * (1 - 0.999) + t * 0.999
