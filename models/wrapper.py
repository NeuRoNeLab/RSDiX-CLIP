import copy

import lightning as l
import numpy as np
import torch
import torch.nn.functional as f
import yaml
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel

from model_utils import sinkhorn, ema, compute_similarities, compute_st_similarities, compute_mse_similarities, \
    get_image_caption_chunks
from utils import CONFIG_DIR, VIT_CONFIG_FILE, MINIBATCH_SIZE, BATCH_SIZE, IMAGE_FIELD, CAPTION_FIELD, BETAS


class CLIPWrapper(l.LightningModule):
    st_model = SentenceTransformer("all-MiniLM-L6-v2")

    def __init__(self, model: str = "openai/clip-vit-base-patch32", batch_size: int = MINIBATCH_SIZE,
                 kl_coeff: float = 1.0, lr: float = None, warmup_steps: int = 0, betas: tuple[float, float] = BETAS,
                 eps: float = 1e-08, weight_decay: float = 0.2):
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

        # changed from minibatch_size to batch_size to match Lightning's BatchSizeFinder expectations
        if batch_size == MINIBATCH_SIZE:
            self._batch_size = BATCH_SIZE
        else:
            self._batch_size = batch_size

        self._warmup_steps = warmup_steps
        self._betas = betas
        self._eps = eps
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

    # necessary to use Lightning's BatchSizeFinder
    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, batch_size):
        self._batch_size = batch_size

    # necessary to use Lightning's LearningRateFinder
    @property
    def lr(self):
        return self._lr

    @lr.setter
    def lr(self, lr):
        self._lr = lr

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
            images_embs[self.global_rank][i * self._batch_size:(i + 1) * self._batch_size] = \
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
            captions_embs = copy.deepcopy(student_caption_embs)
            captions_embs[self.global_rank][i * self._batch_size:(i + 1) * self._batch_size] = \
                f.normalize(self.encode_text(caption_chk), dim=1)
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
        image_chunks, caption_chunks = get_image_caption_chunks(image, caption, self._batch_size)

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

            # teacher
            teacher_image_embs, teacher_captions_embs = self.get_embeddings(image_chunks, caption_chunks, teacher=True)

            sim_ii, sim_tt, sim_it, sim_ti = compute_similarities(torch.cat(teacher_image_embs),
                                                                  torch.cat(teacher_captions_embs))

            # optimal transport
            # Perform sinkhorn based on the cost matrix, and then row-normalize
            # to get target probability.
            img_cost = - (sim_ii + sim_tt + sim_it)
            caption_cost = - (sim_ii + sim_tt + sim_ti)
            img_target = sinkhorn(img_cost)
            caption_target = sinkhorn(caption_cost)

            img_img_sim, img_txt_sim, txt_txt_sim_clip, txt_txt_sim_st \
                = compute_st_similarities(student_images_embs, student_caption_embs, self.st_model.encode(caption))

            loss += self.compute_kl_div(student_image_logits_unscaled, img_target, caption_target)
            self.log_dict({'loss': loss,
                           'mse': compute_mse_similarities(img_img_sim, img_txt_sim, txt_txt_sim_clip, txt_txt_sim_st)},
                          prog_bar=True, on_step=True, on_epoch=True, logger=True, enable_graph=True)

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
        image, caption = batch[IMAGE_FIELD], batch[CAPTION_FIELD]
        image_chunks, caption_chunks = get_image_caption_chunks(image, caption)
        image_embs, caption_embs = self.get_embeddings(image_chunks, caption_chunks)

        image_logits, caption_logits = self.forward(batch)
        ground_truth = torch.arange(len(image_logits), device=batch[IMAGE_FIELD].device)

        img_img_sim, img_txt_sim, txt_txt_sim_clip, txt_txt_sim_st \
            = compute_st_similarities(image_embs, caption_embs, self.st_model.encode(caption))

        loss = (f.cross_entropy(image_logits, ground_truth) + f.cross_entropy(caption_logits, ground_truth)).div(2)
        self.log_dict({'val_loss': loss,
                       'val_mse': compute_mse_similarities(img_img_sim, img_txt_sim, txt_txt_sim_clip, txt_txt_sim_st)},
                      prog_bar=True, on_step=True, on_epoch=True, logger=True, enable_graph=True)

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

        return outputs.logits_per_image, outputs.logits_per_text
        # logits = f.normalize(self.encode_image(image)) @ f.normalize(self.encode_text(caption)).t()
        #
        # return logits, logits.t()  # image logits, caption logits

    def update_teacher(self):
        for teacher, student in zip(self._teacher.parameters(), self._student.parameters()):
            teacher.data.copy_(ema(student.data, teacher.data))

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(params=self._student.parameters(), lr=self._lr, eps=self._eps,
                                      betas=self._betas, weight_decay=self._weight_decay)

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
