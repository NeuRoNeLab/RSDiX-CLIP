import torch

from typing import final
from sentence_transformers.util import cos_sim

REDUCTIONS: final = frozenset(["mean", "average", "avg", "sum", "add", "none"])


# Source: https://github.com/facebookresearch/OTTER/blob/main/models/model_util.py
def _has_nan_or_inf(x):
    return torch.isnan(x).any() or torch.isinf(x).any()


# Source: https://github.com/facebookresearch/OTTER/blob/main/models/model_util.py
@torch.no_grad()
def sinkhorn(cost_mat, eps=0.05, niter=5, r_prob=None, c_prob=None):
    """
    cost_mat: s1, s2, ..., sn, M, N
    r_prob: s1, s2, ..., sn, M
    c_prob: s1, s2, ..., sn, N
    """

    q = torch.exp(-cost_mat / eps)
    q = q / q.sum(dim=[-2, -1], keepdim=True)
    m, n = q.shape[-2], q.shape[-1]

    if r_prob is not None:
        # s1, ..., sn, M -> s1, ..., sn, M, 1
        r_prob = (r_prob / r_prob.sum(dim=-1, keepdim=True)).unsqueeze(-1)
        assert not _has_nan_or_inf(r_prob)
    else:
        r_prob = 1 / m

    if c_prob is not None:
        # s1, ..., sn, N -> s1, ..., sn, 1, N
        c_prob = (c_prob / c_prob.sum(dim=-1, keepdim=True)).unsqueeze(-2)
        assert not _has_nan_or_inf(c_prob)
    else:
        c_prob = 1 / n

    for _ in range(niter):
        # normalize each row: total weight per row must be r_prob
        q /= q.sum(dim=-1, keepdim=True)
        q *= r_prob
        # normalize each column: total weight per column must be c_prob
        q /= q.sum(dim=-2, keepdim=True)
        q *= c_prob
    return q / q.sum(dim=1, keepdim=True)


@torch.no_grad()
def compute_similarities(i_emb, t_emb):
    sim_ii, sim_tt = torch.matmul(i_emb, i_emb.t()), torch.matmul(t_emb, t_emb.t())
    sim_it, sim_ti = torch.matmul(i_emb, t_emb.t()), torch.matmul(t_emb, i_emb.t())
    return sim_ii, sim_tt, sim_it, sim_ti


@torch.no_grad()
def compute_teacher_targets(teacher_images_embs, teacher_text_embs, ii_coeff, tt_coeff, sinkhorn_lambda, sinkhorn_iter,
                            remove_diag):
    sim_ii, sim_tt, sim_it, sim_ti = compute_similarities(teacher_images_embs, teacher_text_embs)

    diag = (torch.eye(*sim_ii.shape) * remove_diag * 1e2).to(teacher_images_embs.device)
    sim_ii = (sim_ii - diag) * ii_coeff
    sim_tt = (sim_tt - diag) * tt_coeff

    # Optimal transport
    # Perform sinkhorn based on the cost matrix, and then row-normalize
    # to get target probability.
    images_cost_mat = - (sim_ii + sim_tt + sim_it)
    text_cost_mat = - (sim_ii + sim_tt + sim_ti)

    images_target_prob = sinkhorn(images_cost_mat, sinkhorn_lambda, sinkhorn_iter)
    text_target_prob = sinkhorn(text_cost_mat, sinkhorn_lambda, sinkhorn_iter)

    images_target_prob /= images_target_prob.sum(dim=1, keepdim=True)
    text_target_prob /= text_target_prob.sum(dim=1, keepdim=True)

    return images_target_prob, text_target_prob


@torch.no_grad()
def compute_st_similarities(clip_image_embeddings, clip_text_embeddings, st_embeddings):
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


@torch.no_grad()
def compute_mse(clip_image_embeddings, clip_text_embeddings, st_embeddings, device):
    ii_sim, it_sim, tt_sim_clip, tt_sim_st = compute_st_similarities(clip_image_embeddings, clip_text_embeddings,
                                                                     st_embeddings)

    if tt_sim_st.device != device:
        tt_sim_st = tt_sim_st.to(device)

    return compute_mse_similarities(ii_sim, it_sim, tt_sim_clip, tt_sim_st)


def compute_accuracy(images_logits: torch.Tensor, batch_size: int):
    ground_truth = torch.arange(len(images_logits)).to(images_logits.device)

    acc_i = (torch.argmax(images_logits, 1) == ground_truth).sum()
    acc_t = (torch.argmax(images_logits, 0) == ground_truth).sum()

    return (acc_i + acc_t) / 2 / batch_size
