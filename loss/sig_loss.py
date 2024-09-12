# Source: https://arxiv.org/abs/2303.15343
import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as nnf


class SigmoidLoss(nn.Module):
    """
    Wrapped SigmoidLoss Loss with a learnable temperature and bias term.
    """
    def __init__(self, temperature_init: float = math.log(10.0), bias_init: float = -10.0):
        super().__init__()
        self.t_s = nn.Parameter(torch.tensor(temperature_init))
        self.bias = nn.Parameter(torch.tensor(bias_init))

    def forward(self, unscaled_logits: torch.Tensor, target: Optional[torch.Tensor] = None) -> torch.Tensor:
        logits = unscaled_logits * self.t_s.exp() + self.bias

        if target is None:
            target = 2 * torch.eye(logits.shape[0], device=unscaled_logits.device) - torch.ones_like(logits)

        loglikelihood = nnf.logsigmoid(target * logits)
        nll = -loglikelihood.sum(dim=-1)
        return nll.mean()
