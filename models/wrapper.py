import copy
from typing import Union

import lightning as l
import numpy as np
import torch
import torch.nn.functional as f
import yaml
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel

from utils import CONFIG_DIR, VIT_CONFIG_FILE, MINIBATCH_SIZE, BATCH_SIZE, IMAGE_FIELD, CAPTION_FIELD, BETAS, \
    RAW_FIELD_CAPTION
from .model_utils import ema, compute_st_similarities, compute_mse_similarities, \
    get_image_caption_chunks, compute_teacher_targets, compute_losses


class CLIPWrapper(l.LightningModule):
    st_model = SentenceTransformer("all-MiniLM-L6-v2")

    def __init__(self, model: str = "openai/clip-vit-base-patch32", batch_size: int = MINIBATCH_SIZE,
                 kl_coeff: float = 1.0, lr: float = None, warmup_steps: int = 0, betas: tuple[float, float] = BETAS,
                 alpha: float = 0.5, eps: float = 1e-08, weight_decay: float = 0.2):
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
        self._alpha = alpha
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

    def compute_training_loss(self, image_chunks, caption_chunks, student_images_embs, student_caption_embs,
                              ground_truth, img_target, caption_target):
        # image loss
        img_contrastive_loss: Union[float, torch.Tensor] = 0.0
        img_distillation_loss: Union[float, torch.Tensor] = 0.0
        for i, img_chk in enumerate(image_chunks):
            # TODO: maybe it's not necessary
            images_embs = [torch.clone(student_images_embs[j]) for j in range(0, len(student_images_embs))]
            images_embs[self.global_rank][i * self._batch_size:(i + 1) * self._batch_size] = \
                f.normalize(self.encode_image(img_chk), dim=1)
            contrastive_loss, distillation_loss = compute_losses(images_embs, student_caption_embs,
                                                                 self._student.logit_scale.detach().item(),
                                                                 ground_truth, img_target, caption_target,
                                                                 self._sink_temp, self._kl_coeff)
            img_contrastive_loss += contrastive_loss
            img_distillation_loss += distillation_loss

        # caption loss
        caption_contrastive_loss: Union[float, torch.Tensor] = 0.0
        caption_distillation_loss: Union[float, torch.Tensor] = 0.0
        for i, caption_chk in enumerate(caption_chunks):
            # TODO: maybe it's not necessary
            captions_embs = [torch.clone(student_caption_embs[j]) for j in range(0, len(student_caption_embs))]
            captions_embs[self.global_rank][i * self._batch_size:(i + 1) * self._batch_size] = \
                f.normalize(self.encode_text(caption_chk), dim=1)
            contrastive_loss, distillation_loss = compute_losses(student_images_embs, captions_embs,
                                                                 self._student.logit_scale.detach().item(),
                                                                 ground_truth, img_target, caption_target,
                                                                 self._sink_temp, self._kl_coeff)

            caption_contrastive_loss += contrastive_loss
            caption_distillation_loss += distillation_loss

        contrastive_loss = img_contrastive_loss + caption_contrastive_loss
        distillation_loss = img_distillation_loss + caption_distillation_loss

        loss = self._alpha * contrastive_loss + (1 - self._alpha) * distillation_loss
        self.manual_backward(loss)

        return loss

    # Training loss: https://github.com/openai/CLIP/issues/83
    # Multi-GPU support: https://github.com/MicPie/clasp
    def training_step(self, batch, batch_idx):
        # get optimizers and lr scheduler
        optimizer = self.optimizers()

        image, caption, raw_caption = batch[IMAGE_FIELD], batch[CAPTION_FIELD], batch[RAW_FIELD_CAPTION]
        image_chunks, caption_chunks = get_image_caption_chunks(image, caption, self._batch_size)

        # calculate original statistics
        with torch.no_grad():
            teacher_image_embs, teacher_captions_embs = self.get_embeddings(image_chunks, caption_chunks,
                                                                            teacher=True)
            img_target, caption_target = compute_teacher_targets(teacher_image_embs, teacher_captions_embs)

        if isinstance(optimizer, list):
            optimizer = optimizer[0]
        optimizer.zero_grad()

        student_images_embs, student_caption_embs = self.get_embeddings(image_chunks, caption_chunks)
        student_image_logits = torch.cat(student_images_embs) @ torch.cat(student_caption_embs).t()
        # contrastive loss ground truth -> Identity matrix
        ground_truth = torch.arange(len(student_image_logits)).long().to(student_image_logits.device)
        loss = self.compute_training_loss(image_chunks, caption_chunks, student_images_embs, student_caption_embs,
                                          ground_truth, img_target, caption_target)

        img_img_sim, img_txt_sim, txt_txt_sim_clip, txt_txt_sim_st \
            = compute_st_similarities(student_images_embs, student_caption_embs,
                                      self.st_model.encode(raw_caption, device=batch[IMAGE_FIELD].device))

        if txt_txt_sim_st.device != batch[IMAGE_FIELD].device:
            txt_txt_sim_st = txt_txt_sim_st.to(batch[IMAGE_FIELD].device)

        acc_i = (torch.argmax(student_image_logits, 1) == ground_truth).sum()
        acc_t = (torch.argmax(student_image_logits, 0) == ground_truth).sum()

        self.log_dict({'loss': loss.item(),
                       'mse_sentence_transformer': compute_mse_similarities(img_img_sim, img_txt_sim, txt_txt_sim_clip,
                                                                            txt_txt_sim_st),
                       'acc': (acc_i + acc_t) / 2 / len(image)},
                      batch_size=self._batch_size,
                      prog_bar=True, on_step=True, on_epoch=True, logger=True, enable_graph=True)

        optimizer.step()
        lr_scheduler = self.lr_schedulers()
        lr_scheduler.step()
        self._student.logit_scale.data.clamp_(-np.log(100), np.log(100))
        self._sink_temp.data.clamp_(-np.log(100), np.log(100))
        self.update_teacher()

        return loss

    def validation_step(self, batch, batch_idx):
        image, caption, raw_caption = batch[IMAGE_FIELD], batch[CAPTION_FIELD], batch[RAW_FIELD_CAPTION]
        image_chunks, caption_chunks = get_image_caption_chunks(image, caption, self._batch_size)
        image_embs, caption_embs = self.get_embeddings(image_chunks, caption_chunks)

        del batch[RAW_FIELD_CAPTION]

        image_logits, caption_logits = self.forward(batch)
        ground_truth = torch.arange(len(image_logits), device=batch[IMAGE_FIELD].device)

        acc_i = (torch.argmax(image_logits, 1) == ground_truth).sum()
        acc_t = (torch.argmax(caption_logits, 1) == ground_truth).sum()

        img_img_sim, img_txt_sim, txt_txt_sim_clip, txt_txt_sim_st \
            = compute_st_similarities(image_embs, caption_embs,
                                      self.st_model.encode(raw_caption, device=batch[IMAGE_FIELD].device))

        if txt_txt_sim_st.device != batch[IMAGE_FIELD].device:
            txt_txt_sim_st = txt_txt_sim_st.to(batch[IMAGE_FIELD].device)

        loss = (f.cross_entropy(image_logits, ground_truth) + f.cross_entropy(caption_logits, ground_truth)).div(2)
        self.log_dict({'val_loss': loss.item(),
                       'val_mse_sentence_transformer': compute_mse_similarities(img_img_sim, img_txt_sim,
                                                                                txt_txt_sim_clip, txt_txt_sim_st),
                       'val_acc': (acc_i + acc_t) / 2 / len(image_logits)},
                      batch_size=self._batch_size,
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
                                                     warmup_steps=min(self._warmup_steps,
                                                                      self.trainer.estimated_stepping_batches - 1))

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler
            }
        }
