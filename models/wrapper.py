import torch
import copy
import math
import yaml

import numpy as np
import lightning as l
import torch.nn.functional as f

from transformers import VisionTextDualEncoderModel
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts


def compute_similarities(i_emb, t_emb):
    sim_ii, sim_tt = i_emb @ t_emb.t(), t_emb @ t_emb.t()
    sim_it, sim_ti = i_emb @ t_emb.t(), t_emb @ t_emb.t()
    return sim_ii, sim_tt, sim_it, sim_ti


def ema(s, t):
    return s * (1 - 0.999) + t * 0.999


def normalize(enc, dim=1):
    return f.normalize(enc, dim=dim)


class CLIPWrapper(l.LightningModule):

    def __init__(self, image_encoder: str, text_encoder: str, batch_size: int = 512, kl_coeff: float = 1.0,
                 learning_rate: float = None, warmup_steps: int = 0, avg_word_embs: bool = False):
        super().__init__()

        self._model = VisionTextDualEncoderModel.from_vision_text_pretained(image_encoder, text_encoder)

        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._warmup_steps = warmup_steps
        self._avg_word_embs = avg_word_embs
        self._sink_temp = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # init self-distillation model
        self._teacher = copy.deepcopy(self._model)
        self._kl_coeff = kl_coeff

        # enable manual_backward
        self.automatic_optimization = False
        # save hyperparameters when checkpointing
        self.save_hyperparameters(ignore=['image_encoder', 'text_encoder'])

    def update_teacher(self):
        for teacher, student in zip(self._teacher.parameters(), self._model.parameters()):
            teacher.data.copy_(ema(student.data, teacher.data))

    # Source: https://github.com/facebookresearch/swav/blob/5e073db0cc69dea22aa75e92bfdd75011e888f28/main_swav.py#L354
    def sinkhorn(self, out):
        Q = torch.exp(out / 0.05).t()  # Q is K-by-B for consistency with notations from our paper
        B = Q.shape[1]  # number of samples to assign
        K = Q.shape[0]  # how many prototypes

        # make the matrix sums to 1
        sum_Q = torch.sum(Q)
        Q /= sum_Q

        for it in range(3):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            Q /= sum_of_rows
            Q /= K

            # normalize each column: total weight per sample must be 1/B
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B  # the columns must sum to 1 so that Q is an assignment
        return Q.t()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(params=self._model.parameters(), lr=self._learning_rate, weight_decay=0.2)

        # Source: https://github.com/PyTorchLightning/pytorch-lightning/issues/5449
        # Source: https://github.com/openai/CLIP/issues/107
        lr_scheduler = CosineAnnealingWarmupRestarts(optimizer,
                                                     first_cycle_steps=self.trainer.estimated_stepping_batches,
                                                     max_lr=self._learning_rate,
                                                     warmup_steps=self._warmup_steps)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler
            }
        }
