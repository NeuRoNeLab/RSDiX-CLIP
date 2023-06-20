import torch
import copy
import math
import yaml

import numpy as np
import lightning as l
import torch.nn.functional as f

from transformers import CLIPModel, CLIPProcessor
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts


def compute_similarities(i_emb, t_emb):
    sim_ii, sim_tt = i_emb @ i_emb.t(), t_emb @ t_emb.t()
    sim_it, sim_ti = i_emb @ t_emb.t(), t_emb @ t_emb.t()
    return sim_ii, sim_tt, sim_it, sim_ti


def ema(s, t):
    return s * (1 - 0.999) + t * 0.999


def normalize(enc, dim=1):
    return f.normalize(enc, dim=dim)


class CLIPWrapper(l.LightningModule):

    def __init__(self, model: str = "openai/clip-vit-base-patch32", processor: str = "openai/clip-vit-base-patch32",
                 batch_size: int = 512, kl_coeff: float = 1.0, lr: float = None,
                 warmup_steps: int = 0, weight_decay: float = 0.2, avg_word_embs: bool = False):
        super().__init__()

        self._model = CLIPModel.from_pretrained(model)
        self._processor = CLIPProcessor.from_pretrained(processor)

        if lr is None:
            model_type = "B" if "base" in model else "L"
            model_size = model.split("patch", 1)[1]
            model_name = f"ViT-{model_type}/{model_size}"

            with open(f"models/configs/ViT.yaml") as stream:
                config = yaml.safe_load(stream)[model_name]

            self._lr = float(config["learning_rate"])
        else:
            self._lr = lr

        self._batch_size = batch_size
        self._warmup_steps = warmup_steps
        self._weight_decay = weight_decay
        self._avg_word_embs = avg_word_embs
        self._sink_temp = torch.nn.Parameter(torch.ones([]) * self._model.logit_scale.item())  # was np.log(1 / 0.07)

        # init self-distillation model
        self._teacher = copy.deepcopy(self._model)
        self._teacher_processor = copy.deepcopy(self._processor)
        self._kl_coeff = kl_coeff

        # enable manual_backward
        self.automatic_optimization = False
        # save hyperparameters when checkpointing
        self.save_hyperparameters(ignore=['image_encoder', 'text_encoder'])

    def _prepare_data(self, img_data, caption_data, teacher=False):
        ims = [f.normalize(self.encode_image(im, teacher=teacher), dim=1) for im in img_data]
        txt = [f.normalize(self.encode_text(t, teacher=teacher), dim=1) for t in caption_data]

        ims, txt = self.all_gather(torch.cat(ims)), self.all_gather(torch.cat(txt))

        if len(ims.shape) == 3:
            ims = list(ims)
            txt = list(txt)
        else:
            ims = [ims]
            txt = [txt]

        return ims, txt

    def _compute_kl_div(self, logit_scores, img_target, txt_target):
        return (f.kl_div(f.log_softmax(logit_scores * self._sink_temp, dim=-1), img_target, reduction='batchmean')
                + f.kl_div(f.log_softmax(logit_scores.t() * self._sink_temp, dim=-1), txt_target,
                           reduction='batchmean')) / 2 * self.kl_coeff

    def _compute_training_loss(self, encode_f, data_mbs, data, ground_truth, txt, img_target, txt_target,
                               is_text=False, ims=None):
        for j, mb in enumerate(data_mbs):
            images_tmp = copy.deepcopy(data)
            images_tmp[self.global_rank][j * self.minibatch_size:(j + 1) * self.minibatch_size] = f.normalize(
                encode_f(mb), dim=1)
            if is_text:
                image_logit_scores_notemp = torch.cat(ims) @ torch.cat(images_tmp).t()
            else:
                image_logit_scores_notemp = torch.cat(images_tmp) @ torch.cat(txt).t()

            image_logit_scores = image_logit_scores_notemp
            loss = (f.cross_entropy(image_logit_scores, ground_truth) + f.cross_entropy(image_logit_scores.t(),
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

            image_logit_scores = torch.cat(ims) @ torch.cat(txt).t()  # scaled logits
            # unscaled logits DO NOT DIFFERENTIATE THROUGH IT
            image_logit_scores_notemp = image_logit_scores / self._model.logit_scale.detach().item()
            ground_truth = torch.arange(len(image_logit_scores)).long().to(image_logit_scores.device)
            loss = (f.cross_entropy(image_logit_scores, ground_truth) + f.cross_entropy(image_logit_scores.t(),
                                                                                        ground_truth)).div(2)
            acc_i = (torch.argmax(image_logit_scores, 1) == ground_truth).sum()
            acc_t = (torch.argmax(image_logit_scores, 0) == ground_truth).sum()
            # calculate teacher
            teacher_ims, teacher_txt = self._prepare_data(image_mbs, text_mbs, teacher=True)

            sim_ii, sim_tt, sim_it, sim_ti = compute_similarities(torch.cat(teacher_ims), torch.cat(teacher_txt))

            # optimal transport
            img_cost = - (sim_ii + sim_tt + sim_it)
            txt_cost = - (sim_ii + sim_tt + sim_ti)
            img_target = self.sinkhorn(img_cost)
            txt_target = self.sinkhorn(txt_cost)
            loss += self._compute_kl_div(image_logit_scores_notemp, img_target, txt_target)
            self.log_dict({'loss': loss / len(ims), 'acc': (acc_i + acc_t) / 2 / len(image) / len(ims)},
                          prog_bar=True, on_step=True, on_epoch=True, logger=True, enable_graph=True)

        if isinstance(optimizer, list):
            optimizer = optimizer[0]
        optimizer.zero_grad()

        # image loss
        self._compute_training_loss(encode_f=self.encode_image, data_mbs=image_mbs, data=ims,
                                    ground_truth=ground_truth, txt=txt, img_target=img_target, txt_target=txt_target)
        # caption loss
        self._compute_training_loss(encode_f=self.encode_text, data_mbs=text_mbs, data=txt,
                                    ground_truth=ground_truth, txt=txt, img_target=img_target, txt_target=txt_target,
                                    is_text=True, ims=ims)

        optimizer.step()
        lr_scheduler = self.lr_schedulers()
        lr_scheduler.step()
        self._model.logit_scale.data.clamp_(-np.log(100), np.log(100))
        self._sink_temp.data.clamp_(-np.log(100), np.log(100))
        self.update_teacher()

    # activates validation loop while training
    def validation_step(self, batch, batch_idx):
        image, caption = batch
        image_logit_scores, caption_logit_scores = self.forward(image, caption)
        ground_truth = torch.arange(len(image_logit_scores), device=image.device)
        loss = (f.cross_entropy(image_logit_scores, ground_truth)
                + f.cross_entropy(caption_logit_scores, ground_truth)).div(2)
        self.log('val_loss', loss)

    def encode_text(self, text, teacher=False):
        inputs = self._processor(text=text, return_tensors="pt") if teacher is False else \
            self._teacher_processor(text=text, return_tensors="pt")

        return self._model.get_text_features(**inputs) if teacher is False else \
            self._teacher.get_text_features(**inputs)

    # def encode_text(self, inputs, teacher=False):
    #     if self.avg_word_embs:
    #         sequence_output = self._teacher.transformer(**inputs)[0] if teacher else self._model.transformer(**inputs)[0]
    #
    #         embeddings = torch.sum(
    #             sequence_output * inputs["attention_mask"].unsqueeze(-1), dim=1
    #         ) / torch.clamp(torch.sum(inputs["attention_mask"], dim=1, keepdim=True), min=1e-9)
    #
    #         return embeddings
    #     else:
    #         return self._teacher.transformer(**inputs)[1] if teacher else self._model.transformer(**inputs)[1]

    def encode_image(self, image, teacher=False):
        inputs = self._processor(image=image, return_tensors="pt") if not teacher else \
            self._teacher_processor(image=image, return_tensors="pt")

        return self._model.get_image_features(**inputs) if not teacher else \
            self._teacher.get_image_features(**inputs)

    def forward(self, image, text):
        logit_scores = normalize(self.encode_image(image)) @ normalize(
            self.encode_text(text)).t()

        return logit_scores, logit_scores.t()

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
        optimizer = torch.optim.AdamW(params=self._model.parameters(), lr=self._lr, weight_decay=self._weight_decay)

        # Source: https://github.com/PyTorchLightning/pytorch-lightning/issues/5449
        # Source: https://github.com/openai/CLIP/issues/107
        lr_scheduler = CosineAnnealingWarmupRestarts(optimizer=optimizer,
                                                     first_cycle_steps=self.trainer.estimated_stepping_batches,
                                                     max_lr=self._lr,
                                                     warmup_steps=self._warmup_steps)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler
            }
        }
