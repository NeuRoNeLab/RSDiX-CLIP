import math
from typing import final

import torch
import torch.nn.functional as f
from sentence_transformers.util import cos_sim

REDUCTIONS: final = frozenset(["mean", "average", "avg", "sum", "add", "none"])


def _has_nan_or_inf(x):
    return torch.isnan(x).any() or torch.isinf(x).any()


@torch.no_grad()
def sinkhorn(cost_mat, eps=0.05, niter=5, r_prob=None, c_prob=None):
    """
    cost_mat: s1, s2, ..., sn, M, N
    r_prob: s1, s2, ..., sn, M
    c_prob: s1, s2, ..., sn, N
    """
    Q = torch.exp(-cost_mat / eps)
    Q = Q / Q.sum(dim=[-2, -1], keepdim=True)
    M, N = Q.shape[-2], Q.shape[-1]

    if r_prob is not None:
        # s1, ..., sn, M -> s1, ..., sn, M, 1
        r_prob = (r_prob / r_prob.sum(dim=-1, keepdim=True)).unsqueeze(-1)
        assert not _has_nan_or_inf(r_prob)
    else:
        r_prob = 1 / M

    if c_prob is not None:
        # s1, ..., sn, N -> s1, ..., sn, 1, N
        c_prob = (c_prob / c_prob.sum(dim=-1, keepdim=True)).unsqueeze(-2)
        assert not _has_nan_or_inf(c_prob)
    else:
        c_prob = 1 / N

    for _ in range(niter):
        # normalize each row: total weight per row must be r_prob
        Q /= Q.sum(dim=-1, keepdim=True)
        Q *= r_prob
        # normalize each column: total weight per column must be c_prob
        Q /= Q.sum(dim=-2, keepdim=True)
        Q *= c_prob
    return Q


# @torch.no_grad()
# Source: https://github.com/facebookresearch/swav/blob/5e073db0cc69dea22aa75e92bfdd75011e888f28/main_swav.py#L354
# def sinkhorn(out):
#     q = torch.exp(out / 0.05).t()  # q is k-by-b for consistency with notations from our paper
#     b = q.shape[1]  # number of samples to assign
#     k = q.shape[0]  # how many prototypes
#
#     # make the matrix sums to 1
#     sum_q = torch.sum(q)
#     q /= sum_q
#
#     for it in range(3):
#         # normalize each row: total weight per prototype must be 1/k
#         sum_of_rows = torch.sum(q, dim=1, keepdim=True)
#         q /= sum_of_rows
#         q /= k
#
#         # normalize each column: total weight per sample must be 1/b
#         q /= torch.sum(q, dim=0, keepdim=True)
#         q /= b
#
#     q *= b  # the columns must sum to 1 so that q is an assignment
#     return q.t()


@torch.no_grad()
def compute_similarities(i_emb, t_emb):
    sim_ii, sim_tt = i_emb @ i_emb.t(), t_emb @ t_emb.t()
    sim_it, sim_ti = i_emb @ t_emb.t(), t_emb @ i_emb.t()
    return sim_ii, sim_tt, sim_it, sim_ti


@torch.no_grad()
def compute_teacher_targets(teacher_image_embs, teacher_captions_embs):
    sim_ii, sim_tt, sim_it, sim_ti = compute_similarities(torch.cat(teacher_image_embs),
                                                          torch.cat(teacher_captions_embs))

    # optimal transport
    # Perform sinkhorn based on the cost matrix, and then row-normalize
    # to get target probability.
    img_cost = - (sim_ii + sim_tt + sim_it)
    caption_cost = - (sim_ii + sim_tt + sim_ti)
    img_target, caption_target = sinkhorn(img_cost), sinkhorn(caption_cost)

    img_target /= img_target.sum(dim=1, keepdim=True)
    caption_target /= caption_target.sum(dim=1, keepdim=True)

    return img_target, caption_target


@torch.no_grad()
def compute_st_similarities(clip_image_embeddings, clip_text_embeddings, st_embeddings):
    if isinstance(clip_image_embeddings, list):
        if len(clip_image_embeddings) > 1:
            clip_image_embeddings = torch.stack(clip_image_embeddings)
        else:
            clip_image_embeddings = clip_image_embeddings[0]

    if isinstance(clip_text_embeddings, list):
        if len(clip_text_embeddings) > 1:
            clip_text_embeddings = torch.stack(clip_text_embeddings)
        else:
            clip_text_embeddings = clip_text_embeddings[0]

    image_image_similarities = cos_sim(clip_image_embeddings, clip_image_embeddings)
    image_text_similarities = cos_sim(clip_image_embeddings, clip_text_embeddings)
    text_text_similarities_clip = cos_sim(clip_text_embeddings, clip_text_embeddings)
    text_text_similarities_st = cos_sim(st_embeddings, st_embeddings)

    return image_image_similarities, image_text_similarities, text_text_similarities_clip, text_text_similarities_st


@torch.no_grad()
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


def compute_losses(image_embs, caption_embs, scale, ground_truth, img_target, caption_target, sink_temp, kl_coeff,
                   reduction="batchmean"):
    logits_unscaled = torch.cat(image_embs) @ torch.cat(caption_embs).t()
    scale = scale.exp()
    sink_temp = sink_temp.exp()
    logits = logits_unscaled * scale

    contrastive_loss = (f.cross_entropy(logits, ground_truth) + f.cross_entropy(logits.t(), ground_truth)) / 2
    distillation_loss = (f.kl_div(f.log_softmax(logits_unscaled * sink_temp, dim=-1), img_target, reduction=reduction) +
                         f.kl_div(f.log_softmax(logits_unscaled.t() * sink_temp, dim=-1), caption_target,
                                  reduction=reduction)) / 2 * kl_coeff

    return contrastive_loss, distillation_loss


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
