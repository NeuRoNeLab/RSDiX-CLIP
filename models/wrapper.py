import copy
import math
import torch
import yaml

import numpy as np
import pytorch_lightning as pl
import torch.nn.functional as F

from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from .model import CLIP


def compute_similarities(i_emb, t_emb):
    sim_ii, sim_tt = i_emb @ t_emb.t(), t_emb @ t_emb.t()
    sim_it, sim_ti = i_emb @ t_emb.t(), t_emb @ t_emb.t()
    return sim_ii, sim_tt, sim_it, sim_ti


def ema(s, t):
    return s * (1 - 0.999) + t * 0.999


def normalize(enc, dim=1):
    return F.normalize(enc, dim=dim)


class CustomCLIPWrapper(pl.LightningModule):
    def __init__(self, model_name: str = "RN50", image_encoder=None, text_encoder=None, minibatch_size: int = 512,
                 avg_word_embs: bool = False):
        super().__init__()

        self.isVit = "ViT" in model_name

        with open(f"models/configs/{'Vit' if self.isVit else 'RN'}.yaml") as stream:
            config = yaml.safe_load(stream)[model_name]

        self.config = copy.deepcopy(config)
        self.config["learning_rate"] = float(self.config["learning_rate"])
        del config["learning_rate"]
        self.model = CLIP(**config)

        if image_encoder:
            del self.model.visual
            self.model.visual = image_encoder

        if text_encoder:
            del self.model.transformer
            self.model.transformer = text_encoder

        self.minibatch_size = minibatch_size
        self.avg_word_embs = avg_word_embs

        # init self-distillation model
        self.teacher = copy.deepcopy(self.model)

    def _prepare_data(self, img_data, caption_data, teacher=False):
        ims = [F.normalize(self.model.encode_image(im), dim=1) for im in img_data]
        txt = [F.normalize(self.encode_text(t, teacher=teacher), dim=1) for t in caption_data]

        ims, txt = self.all_gather(torch.cat(ims)), self.all_gather(torch.cat(txt))

        if len(ims.shape) == 3:
            ims = list(ims)
            txt = list(txt)
        else:
            ims = [ims]
            txt = [txt]

        return ims, txt

    def _compute_kl_div(self, logit_scores, img_target, txt_target):
        return (F.kl_div(F.log_softmax(logit_scores * self.sink_temp, dim=-1), img_target, reduction='batchmean')
                + F.kl_div(F.log_softmax(logit_scores.t() * self.sink_temp, dim=-1), txt_target,
                           reduction='batchmean')) / 2 * self.kl_coeff

    def _compute_training_loss(self, encode_f, data_mbs, data, ground_truth, txt, img_target, txt_target,
                               is_text=False, ims=None):
        for j, mb in enumerate(data_mbs):
            images_tmp = copy.deepcopy(data)
            images_tmp[self.global_rank][j * self.minibatch_size:(j + 1) * self.minibatch_size] = F.normalize(
                encode_f(mb), dim=1)
            if is_text:
                image_logit_scores_notemp = torch.cat(ims) @ torch.cat(images_tmp).t()
            else:
                image_logit_scores_notemp = torch.cat(images_tmp) @ torch.cat(txt).t()
            image_logit_scores = image_logit_scores_notemp * self.model.logit_scale.exp()
            loss = (F.cross_entropy(image_logit_scores, ground_truth) + F.cross_entropy(image_logit_scores.t(),
                                                                                        ground_truth)) / 2
            loss += self._compute_kl_div(image_logit_scores_notemp, img_target, txt_target)
            self.manual_backward(loss)

    # activates training loop
    # Training loss: https://github.com/openai/CLIP/issues/83
    # Multi-GPU support: https://github.com/MicPie/clasp
    def training_step(self, batch, batch_idx):
        # get optimizers and scheduler
        optimizer = self.optimizers()

        image, caption = batch
        n = math.ceil(len(image) // self.minibatch_size)
        image_mbs = torch.chunk(image, n)
        caption_mbs_ids = torch.chunk(torch.arange(len(image)), n)

        # adjust embedding dictionaries
        text_mbs = []
        for s in caption_mbs_ids:
            d = {}
            for key in list(caption.keys()):
                d[key] = caption[key][s]
            text_mbs.append(d)

        # calculate original statistics
        with torch.no_grad():
            ims, txt = self._prepare_data(image_mbs, text_mbs)

            image_logit_scores_notemp = torch.cat(ims) @ torch.cat(txt).t()
            image_logit_scores = image_logit_scores_notemp * self.model.logit_scale.exp()
            ground_truth = torch.arange(len(image_logit_scores)).long().to(image_logit_scores.device)
            loss = (F.cross_entropy(image_logit_scores, ground_truth) + F.cross_entropy(image_logit_scores.t(),
                                                                                        ground_truth)).div(2)
            acc_i = (torch.argmax(image_logit_scores, 1) == ground_truth).sum()
            acc_t = (torch.argmax(image_logit_scores, 0) == ground_truth).sum()
            # calculate teacher
            teacher_ims, teacher_txt = self._prepare_data(image_mbs, text_mbs, teacher=True)

            sim_ii, sim_tt, sim_it, sim_ti = self.compute_similarities(torch.cat(teacher_ims),
                                                                       torch.cat(teacher_txt))

            # optimal transport
            img_cost = - (sim_ii + sim_tt + sim_it)
            txt_cost = - (sim_ii + sim_tt + sim_ti)
            img_target = self.sinkhorn(img_cost)
            txt_target = self.sinkhorn(txt_cost)
            loss += self._compute_kl_div(image_logit_scores_notemp, img_target, txt_target)
            self.log_dict({'loss': loss / len(ims), 'acc': (acc_i + acc_t) / 2 / len(image) / len(ims)},
                          prog_bar=True)

        if isinstance(optimizer, list):
            optimizer = optimizer[0]
        optimizer.zero_grad()

        # image loss
        self._compute_training_loss(self.model.encode_image, image_mbs, ims, ground_truth, txt, img_target,
                                    txt_target)
        # caption loss
        self._compute_training_loss(self.model.encode_text, text_mbs, txt, ground_truth, txt, img_target,
                                    txt_target,
                                    True, ims)

        optimizer.step()
        lr_scheduler = self.lr_schedulers()
        lr_scheduler.step()
        self.model.logit_scale.data.clamp_(-np.log(100), np.log(100))
        self.sink_temp.data.clamp_(-np.log(100), np.log(100))
        self.update_teacher()

    # Source: https://github.com/PyTorchLightning/pytorch-lightning/issues/5449
    def training_steps_nums(self) -> int:
        dataset_size = len(self.trainer.train_dataloader)
        num_devices = max(1, self.trainer.num_devices)

        return (dataset_size * self.trainer.max_epochs) // (self.trainer.accumulate_grad_batches * num_devices)

    # activates validation loop while training
    def validation_step(self, batch, batch_idx):
        image, caption = batch
        image_logit_scores, caption_logit_scores = self.forward(image, caption)
        ground_truth = torch.arange(len(image_logit_scores))
        loss = (F.cross_entropy(image_logit_scores, ground_truth)
                + F.cross_entropy(caption_logit_scores, ground_truth)).div(2)
        self.log('val_loss', loss)

    def encode_text(self, inputs, teacher=False):
        if self.avg_word_embs:
            sequence_output = self.teacher.transformer(**inputs)[0] if teacher else self.model.transformer(**inputs)[0]

            embeddings = torch.sum(
                sequence_output * inputs["attention_mask"].unsqueeze(-1), dim=1
            ) / torch.clamp(torch.sum(inputs["attention_mask"], dim=1, keepdim=True), min=1e-9)

            return embeddings
        else:
            return self.teacher.transformer(**inputs)[1] if teacher else self.model.transformer(**inputs)[1]

    def forward(self, images, text):
        logit_scores = normalize(self.model.encode_image(images)) @ normalize(
            self.encode_text(text)).t() * self.model.logit_scale.exp()

        return logit_scores, logit_scores.t()

    def update_teacher(self):
        for teacher, student in zip(self.teacher.parameters(), self.model.parameters()):
            teacher.data.copy_(self.ema(student.data, teacher.data))

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

    def configure_optimizers(self) -> dict:
        self.trainer.fit_loop.setup_data()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config["learning_rate"],
                                      betas=(0.9, 0.98 if self.isVit else 0.999), eps=1e-6 if self.isVit else 1e-8,
                                      weight_decay=0.2)

        # Source: https://github.com/openai/CLIP/issues/107
        lr_scheduler = CosineAnnealingWarmupRestarts(optimizer, first_cycle_steps=self.training_steps_nums(),
                                                     max_lr=self.config["learning_rate"])

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler
            }
        }
