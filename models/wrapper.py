import copy

import lightning as l
import torch
import torch.nn.functional as f
import yaml
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from sentence_transformers import SentenceTransformer
from torch.optim.lr_scheduler import LinearLR
from transformers import CLIPModel

from loss import DistillationLoss
from utils import CONFIG_DIR, VIT_CONFIG_FILE, MINIBATCH_SIZE, BATCH_SIZE, IMAGE_FIELD, CAPTION_FIELD, BETAS, \
    RAW_FIELD_CAPTION
from .model_utils import ema, compute_st_similarities, compute_mse_similarities, \
    compute_teacher_targets


class CLIPWrapper(l.LightningModule):
    st_model = SentenceTransformer("all-MiniLM-L6-v2")

    def __init__(self, model: str = "openai/clip-vit-base-patch32", batch_size: int = MINIBATCH_SIZE,
                 kl_coeff: float = 1.0, lr: float = None, use_warmup: str = "cosine", warmup_steps: int = 0,
                 betas: tuple[float, float] = BETAS, alpha: float = 0.5, eps: float = 1e-08, weight_decay: float = 0.2,
                 start_factor: float = 0.3333333333333333, end_factor: float = 1.0, total_iters: int = 5):
        super().__init__()

        if use_warmup != "cosine" and use_warmup != "linear":
            raise Exception(f"{use_warmup} not supported. Try 'cosine' or 'linear'. ")

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

        self._use_warmup = use_warmup
        self._weight_decay = weight_decay
        self._betas = betas
        self._alpha = alpha
        self._eps = eps

        if use_warmup == "cosine":
            self._warmup_steps = warmup_steps
        else:
            self._start_factor = start_factor
            self._end_factor = end_factor
            self._total_iters = total_iters
        # Source: https://lightning.ai/docs/pytorch/stable/accelerators/accelerator_prepare.html
        self.register_buffer("_sink_temp", torch.nn.Parameter(torch.ones([]) * self._student.logit_scale.item()))

        # init self-distillation model
        self.dist_loss = DistillationLoss()
        self._teacher = copy.deepcopy(self._student)

        self._kl_coeff = kl_coeff

        # save hyperparameters when checkpointing
        self.save_hyperparameters()

    # necessary to use Lightning's LearningRateFinder
    @property
    def lr(self):
        return self._lr

    @lr.setter
    def lr(self, lr):
        self._lr = lr

    def get_embeddings(self, images, captions, teacher=False):
        images_embeds = self.encode_image(image=images, teacher=teacher)
        captions_embeds = self.encode_text(caption=captions, teacher=teacher)

        # normalized features
        images_embeds = images_embeds / images_embeds.norm(p=2, dim=-1, keepdim=True)
        captions_embeds = captions_embeds / captions_embeds.norm(p=2, dim=-1, keepdim=True)

        return images_embeds, captions_embeds

    # Training loss: https://github.com/openai/CLIP/issues/83
    # Multi-GPU support: https://github.com/MicPie/clasp
    def training_step(self, batch, batch_idx):
        # get optimizers and lr scheduler
        optimizer = self.optimizers()

        images, captions, raw_captions = batch[IMAGE_FIELD], batch[CAPTION_FIELD], batch[RAW_FIELD_CAPTION]

        del batch[RAW_FIELD_CAPTION]

        # calculate original statistics
        with torch.no_grad():
            teacher_image_embs, teacher_captions_embs = self.get_embeddings(images, captions, teacher=True)
            img_target, caption_target = compute_teacher_targets([teacher_image_embs], [teacher_captions_embs])

        if isinstance(optimizer, list):
            optimizer = optimizer[0]
        optimizer.zero_grad()

        contrastive_loss = self.forward(batch).loss

        student_images_embs, student_caption_embs = self.get_embeddings(images, captions)
        student_image_logits_unscaled = torch.cat([student_images_embs]) @ torch.cat([student_caption_embs]).t()
        sink_temp = self._sink_temp.exp()

        # contrastive loss ground truth -> Identity matrix
        ground_truth = torch.arange(len(student_image_logits_unscaled)).long().to(student_image_logits_unscaled.device)
        # distillation_loss = (f.kl_div(f.log_softmax(student_image_logits_unscaled * sink_temp, dim=-1),
        #                               img_target,
        #                               reduction="batchmean") +
        #                      f.kl_div(f.log_softmax(student_image_logits_unscaled.t() * sink_temp, dim=-1),
        #                               caption_target,
        #                               reduction="batchmean")) / 2 * self._kl_coeff
        img_dist_loss = self.dist_loss(pred=student_image_logits_unscaled, target=img_target)
        txt_dist_loss = self.dist_loss(pred=student_image_logits_unscaled.t(), target=caption_target)

        distillation_loss = img_dist_loss + txt_dist_loss

        img_img_sim, img_txt_sim, txt_txt_sim_clip, txt_txt_sim_st \
            = compute_st_similarities(student_images_embs, student_caption_embs,
                                      self.st_model.encode(raw_captions, device=batch[IMAGE_FIELD].device))

        if txt_txt_sim_st.device != batch[IMAGE_FIELD].device:
            txt_txt_sim_st = txt_txt_sim_st.to(batch[IMAGE_FIELD].device)

        acc_i = (torch.argmax(student_image_logits_unscaled, 1) == ground_truth).sum()
        acc_t = (torch.argmax(student_image_logits_unscaled, 0) == ground_truth).sum()

        loss = self._alpha * contrastive_loss + (1 - self._alpha) * distillation_loss

        self.log_dict({'loss': loss.item(),
                       'mse_sentence_transformer': compute_mse_similarities(img_img_sim, img_txt_sim, txt_txt_sim_clip,
                                                                            txt_txt_sim_st),
                       'acc': (acc_i + acc_t) / 2 / len(images)},
                      prog_bar=True, on_step=True, on_epoch=True, logger=True, enable_graph=True)

        self.update_teacher()

        return loss

    def validation_step(self, batch, batch_idx):
        images, captions, raw_captions = batch[IMAGE_FIELD], batch[CAPTION_FIELD], batch[RAW_FIELD_CAPTION]
        image_embs, caption_embs = self.get_embeddings(images, captions)

        del batch[RAW_FIELD_CAPTION]

        outputs = self.forward(batch)
        image_logits, caption_logits = outputs.logits_per_image, outputs.logits_per_text
        ground_truth = torch.arange(len(image_logits), device=batch[IMAGE_FIELD].device)

        acc_i = (torch.argmax(image_logits, 1) == ground_truth).sum()
        acc_t = (torch.argmax(caption_logits, 1) == ground_truth).sum()

        img_img_sim, img_txt_sim, txt_txt_sim_clip, txt_txt_sim_st \
            = compute_st_similarities(image_embs, caption_embs,
                                      self.st_model.encode(raw_captions, device=batch[IMAGE_FIELD].device))

        if txt_txt_sim_st.device != batch[IMAGE_FIELD].device:
            txt_txt_sim_st = txt_txt_sim_st.to(batch[IMAGE_FIELD].device)

        self.log_dict({'val_loss': outputs.loss.item(),
                       'val_mse_sentence_transformer': compute_mse_similarities(img_img_sim, img_txt_sim,
                                                                                txt_txt_sim_clip, txt_txt_sim_st),
                       'val_acc': (acc_i + acc_t) / 2 / len(image_logits)},
                      prog_bar=True, on_step=True, on_epoch=True, logger=True, enable_graph=True)

    def encode_image(self, image, teacher=False):
        # image = inputs["pixel_values"]
        return self._student.get_image_features(image) if teacher is False else \
            self._teacher.get_image_features(image)

    def encode_text(self, caption, teacher=False):
        # caption = inputs["input_ids"]
        return self._student.get_text_features(caption) if teacher is False else \
            self._teacher.get_text_features(caption)

    def forward(self, inputs, return_loss: bool = True):
        return self._student(**inputs, return_loss=return_loss)

    def update_teacher(self):
        for teacher, student in zip(self._teacher.parameters(), self._student.parameters()):
            teacher.data.copy_(ema(student.data, teacher.data))

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(params=self._student.parameters(), lr=self._lr, eps=self._eps,
                                      betas=self._betas, weight_decay=self._weight_decay)

        # Source: https://github.com/PyTorchLightning/pytorch-lightning/issues/5449
        # Source: https://github.com/openai/CLIP/issues/107
        if self._use_warmup == "cosine":
            lr_scheduler = CosineAnnealingWarmupRestarts(optimizer=optimizer,
                                                         first_cycle_steps=self.trainer.estimated_stepping_batches,
                                                         max_lr=self._lr,
                                                         warmup_steps=min(self._warmup_steps,
                                                                          self.trainer.estimated_stepping_batches - 1))
        else:
            lr_scheduler = LinearLR(optimizer, start_factor=self._start_factor, end_factor=self._end_factor,
                                    total_iters=self._total_iters)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler
            }
        }
