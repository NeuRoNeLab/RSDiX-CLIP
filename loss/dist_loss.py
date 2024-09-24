# Source: https://github.com/facebookresearch/OTTER/blob/main/loss/kl_div_loss.py#L10

import torch
import torch.nn as nn


class DistillationLoss(nn.Module):
    """
    Wrapped KLDivergence Loss with a learnable temperature.
    """
    def __init__(self):
        super().__init__()
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.loss = nn.KLDivLoss(reduction='batchmean')
        self.T_s = nn.Parameter(torch.tensor(3.9, requires_grad=True))

    def forward(self, unscaled_logits, target):
        """
        Pred is logits and target is probabilities.
        """
        T_s = torch.clamp(torch.exp(self.T_s), min=1.0, max=100.0)
        pred_logprob = self.logsoftmax(unscaled_logits * T_s)

        return self.loss(input=pred_logprob, target=target)