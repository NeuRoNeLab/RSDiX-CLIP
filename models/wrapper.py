import torch
import copy
import math
import yaml

import numpy as np
import lightning as l
import torch.nn.functional as f

from utils import CONFIG_DIR, VIT_CONFIG_FILE, MINIBATCH_SIZE, BATCH_SIZE, IMAGE_FIELD, CAPTION_FIELD
from transformers import CLIPModel
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts


def compute_similarities(i_emb, t_emb):
    sim_ii, sim_tt = i_emb @ i_emb.t(), t_emb @ t_emb.t()
    sim_it, sim_ti = i_emb @ t_emb.t(), t_emb @ t_emb.t()
    return sim_ii, sim_tt, sim_it, sim_ti


def ema(s, t):
    return s * (1 - 0.999) + t * 0.999


class CLIPWrapper(l.LightningModule):

    def __init__(self, model: str = "openai/clip-vit-base-patch32", minibatch_size: int = MINIBATCH_SIZE,
                 kl_coeff: float = 1.0, lr: float = None, warmup_steps: int = 0, weight_decay: float = 0.2):
        super().__init__()

        self._student = CLIPModel.from_pretrained(model)

        if lr is None:
            model_type = "B" if "base" in model else "L"
            model_size = model.split("patch", 1)[1]
            model_name = f"ViT-{model_type}/{model_size}"

            with open(f"{CONFIG_DIR}/{VIT_CONFIG_FILE}") as stream:
                config = yaml.safe_load(stream)[model_name]

            self._lr = float(config["learning_rate"])
        else:
            self._lr = lr

        if minibatch_size == MINIBATCH_SIZE:
            self._minibatch_size = BATCH_SIZE
        else:
            self._minibatch_size = minibatch_size

        self._warmup_steps = warmup_steps
        self._weight_decay = weight_decay
        # Source: https://lightning.ai/docs/pytorch/stable/accelerators/accelerator_prepare.html
        self.register_buffer("_sink_temp", torch.nn.Parameter(torch.ones([]) * self._student.logit_scale.item()))

        # init self-distillation model
        self._teacher = copy.deepcopy(self._student)

        self._kl_coeff = kl_coeff

        # enable manual backward
        self.automatic_optimization = False

        # save hyperparameters when checkpointing
        self.save_hyperparameters(ignore=["image_encoder", "text_encoder"])

    def get_embeddings(self, images, captions, teacher=False):
        image_embs = [f.normalize(self.encode_image(image, teacher=teacher), dim=1) for image in images]
        caption_embs = [f.normalize(self.encode_text(caption, teacher=teacher), dim=1) for caption in captions]

        # sync and gather data from all devices
        image_embs, caption_embs = self.all_gather(torch.cat(image_embs)), self.all_gather(torch.cat(caption_embs))

        if len(image_embs.shape) == 3:
            image_embs = list(image_embs)
            caption_embs = list(caption_embs)
        else:
            image_embs = [image_embs]
            caption_embs = [caption_embs]

        return image_embs, caption_embs

    def compute_kl_div(self, logit_scores, img_target, caption_target, reduction="batchmean"):
        return (f.kl_div(f.log_softmax(logit_scores * self._sink_temp, dim=-1), img_target, reduction=reduction) +
                f.kl_div(f.log_softmax(logit_scores.t() * self._sink_temp, dim=-1), caption_target,
                         reduction=reduction)) / 2 * self._kl_coeff

    def compute_training_loss(self, image_chunks, caption_chunks, student_images_embs, student_caption_embs,
                              ground_truth, img_target, caption_target):
        # image loss
        for i, img_chk in enumerate(image_chunks):
            # TODO: maybe its not necessary
            images_embs = copy.deepcopy(student_images_embs)
            images_embs[self.global_rank][i * self._minibatch_size:(i + 1) * self._minibatch_size] = \
                f.normalize(self.encode_image(img_chk), dim=1)
            # scaled logits with self.student.logit_scale()
            image_logits = torch.cat(images_embs) @ torch.cat(student_caption_embs).t()
            # unscaled logits DO NOT DIFFERENTIATE THROUGH IT
            image_logits_unscaled = image_logits / self._student.logit_scale.detach().item()
            loss = (f.cross_entropy(image_logits, ground_truth) + f.cross_entropy(image_logits.t(),
                                                                                  ground_truth)) / 2
            loss += self.compute_kl_div(image_logits_unscaled, img_target, caption_target)
            self.manual_backward(loss)

        # caption loss
        for i, caption_chk in enumerate(caption_chunks):
            # TODO: maybe its not necessary
            captions_embs = copy.deepcopy(student_images_embs)
            captions_embs[self.global_rank][i * self._minibatch_size:(i + 1) * self._minibatch_size] = \
                f.normalize(self.encode_image(caption_chk), dim=1)
            # scaled logits with self.student.logit_scale()
            caption_logits = torch.cat(student_images_embs) @ torch.cat(captions_embs).t()
            # unscaled logits DO NOT DIFFERENTIATE THROUGH IT
            caption_logits_unscaled = caption_logits / self._student.logit_scale.detach().item()
            loss = (f.cross_entropy(caption_logits, ground_truth) + f.cross_entropy(caption_logits.t(),
                                                                                    ground_truth)) / 2
            loss += self.compute_kl_div(caption_logits_unscaled, img_target, caption_target)
            self.manual_backward(loss)

    # Training loss: https://github.com/openai/CLIP/issues/83
    # Multi-GPU support: https://github.com/MicPie/clasp
    def training_step(self, batch, batch_idx):
        # get optimizers and lr scheduler
        optimizer = self.optimizers()

        image, caption = batch[IMAGE_FIELD], batch[CAPTION_FIELD]
        n = math.ceil(len(image) // self._minibatch_size)
        image_chunks = torch.chunk(image, n)
        caption_chunks_ids = torch.chunk(torch.arange(len(image)), n)

        # adjust embedding dictionaries
        caption_chunks = []
        for s in caption_chunks_ids:
            d = {}
            for key in list(caption.keys()):
                d[key] = caption[key][s]
            caption_chunks.append(d)

        # calculate original statistics
        with torch.no_grad():
            # student
            student_images_embs, student_caption_embs = self.get_embeddings(image_chunks, caption_chunks)

            # scaled logits with self.student.logit_scale()
            # TODO: check if self.forward is equivalent
            student_image_logits = torch.cat(student_images_embs) @ torch.cat(student_caption_embs).t()
            # unscaled logits DO NOT DIFFERENTIATE THROUGH IT
            student_image_logits_unscaled = student_image_logits / self._student.logit_scale.detach().item()
            # contrastive loss ground truth -> Identity matrix
            ground_truth = torch.arange(len(student_image_logits)).long().to(student_image_logits.device)
            loss = (f.cross_entropy(student_image_logits, ground_truth) + f.cross_entropy(student_image_logits.t(),
                                                                                          ground_truth)).div(2)

            acc_i = (torch.argmax(student_image_logits, 1) == ground_truth).sum()
            acc_t = (torch.argmax(student_image_logits, 0) == ground_truth).sum()

            # teacher
            teacher_image_embs, teacher_captions_embs = self.get_embeddings(image_chunks, caption_chunks, teacher=True)

            sim_ii, sim_tt, sim_it, sim_ti = compute_similarities(torch.cat(teacher_image_embs),
                                                                  torch.cat(teacher_captions_embs))

            # optimal transport
            img_cost = - (sim_ii + sim_tt + sim_it)
            caption_cost = - (sim_ii + sim_tt + sim_ti)
            img_target = self.sinkhorn(img_cost)
            caption_target = self.sinkhorn(caption_cost)

            loss += self.compute_kl_div(student_image_logits_unscaled, img_target, caption_target)
            self.log_dict({'loss': loss / len(student_images_embs),
                           'acc': (acc_i + acc_t) / 2 / len(image) / len(student_images_embs)}, prog_bar=True,
                          on_step=True, on_epoch=True, logger=True, enable_graph=True)

            if isinstance(optimizer, list):
                optimizer = optimizer[0]
            optimizer.zero_grad()

            self.compute_training_loss(image_chunks, caption_chunks, student_images_embs, student_caption_embs,
                                       ground_truth, img_target, caption_target)

            optimizer.step()
            lr_scheduler = self.lr_schedulers()
            lr_scheduler.step()
            self._student.logit_scale.data.clamp_(-np.log(100), np.log(100))
            self._sink_temp.data.clamp_(-np.log(100), np.log(100))
            self.update_teacher()

    def validation_step(self, batch, batch_idx):
        image_logits, caption_logits = self.forward(batch)
        ground_truth = torch.arange(len(image_logits), device=batch[IMAGE_FIELD].device)

        acc_i = (torch.argmax(image_logits, 1) == ground_truth).sum()
        acc_t = (torch.argmax(caption_logits, 1) == ground_truth).sum()

        loss = (f.cross_entropy(image_logits, ground_truth) + f.cross_entropy(caption_logits, ground_truth)).div(2)
        self.log_dict({'val_loss': loss / len(image_logits),
                       'acc': (acc_i + acc_t) / 2 / len(image_logits) / len(image_logits)}, prog_bar=True,
                      on_step=True, on_epoch=True, logger=True, enable_graph=True)

    def encode_image(self, image, teacher=False):
        # image = inputs["pixel_values"]
        return self._student.get_image_features(image) if teacher is False else \
            self._teacher.get_image_features(image)

    def encode_text(self, caption, teacher=False):
        # caption = inputs["input_ids"]
        return self._student.get_text_features(caption) if teacher is False else \
            self._teacher.get_text_features(caption)

    def forward(self, inputs):
        outputs = self._student(**inputs)

        return outputs.logits_per_image, outputs.logits_per_image.t()
        # logits = f.normalize(self.encode_image(image)) @ f.normalize(self.encode_text(caption)).t()
        #
        # return logits, logits.t()  # image logits, caption logits

    def update_teacher(self):
        for teacher, student in zip(self._teacher.parameters(), self._student.parameters()):
            teacher.data.copy_(ema(student.data, teacher.data))

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

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(params=self._student.parameters(), lr=self._lr, weight_decay=self._weight_decay)

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
