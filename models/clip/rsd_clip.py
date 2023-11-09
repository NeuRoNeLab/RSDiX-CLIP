import lightning as l
import torch
import yaml
import copy

from transformers import CLIPModel
from typing import Optional

from loss import DistillationLoss
from sentence_transformers import SentenceTransformer
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from torch.optim.lr_scheduler import LinearLR

from utils import CONFIG_DIR, VIT_CONFIG_FILE, IMAGE_FIELD, CAPTION_FIELD, BETAS, \
    RAW_CAPTION_FIELD
from .model_utils import compute_mse, compute_accuracy, compute_teacher_targets
from .ema import ExponentialMovingAverage


class RSDClip(l.LightningModule):
    """
    A Pytorch-Lightning based wrapper for Hugging Face's CLIP implementation.
    """

    _st_model = SentenceTransformer("all-MiniLM-L6-v2")

    def __init__(self,
                 model: str = "openai/clip-vit-base-patch32",
                 lr: Optional[float] = None,
                 alpha: float = 0.5,
                 ema_decay: float = 0.999,
                 weight_decay: float = 0.1,
                 start_factor: float = 0.3333333333333333,
                 end_factor: float = 1.0,
                 total_iters: int = 5,
                 use_warmup: str = "cosine",
                 warmup_steps: int = 0,
                 eps: float = 1e-08,
                 betas: tuple[float, float] = BETAS,
                 sinkhorn_lambda: float = 0.1,
                 sinkhorn_iter: int = 4,
                 ii_coeff: float = 1.0,
                 tt_coeff: float = 1.0,
                 remove_diag: bool = False):
        """
            Initialize a CLIPWrapper instance.

            Parameters:
                model (str): The pre-trained CLIP model to use. Defaults to "openai/clip-vit-base-patch32".
                lr (Optional[float]): The learning rate for the optimizer. If not provided, it attempts to use the
                    learning rate from the model's configuration.
                alpha (float): Trade-off factor between the contrastive loss and self-distillation loss.
                    Defaults to 0.5 for equal contributions from both losses.
                ema_decay (float): Exponential Moving Average (EMA) decay factor for the teacher model. Controls the
                    adaptation rate of the teacher model.
                weight_decay (float): Weight decay applied to model parameters during optimization.
                start_factor (float): Starting factor for the learning rate schedule during linear warm-up.
                end_factor (float): Ending factor for the learning rate schedule during linear warm-up.
                total_iters (int): Total number of iterations over which linear warm-up is applied.
                use_warmup (str): Specifies whether to use warm-up for learning rate scheduling.
                    Choose between "cosine" or "linear."
                warmup_steps (int): Number of warm-up steps.
                eps (float): A small epsilon value added to prevent division by zero when normalizing embeddings.
                betas (tuple[float, float]): Beta coefficients for the Adam optimizer.
                    Control exponential moving averages of gradient and squared gradient.
                sinkhorn_lambda (float): Parameter used in Sinkhorn distance computation for self-distillation.
                sinkhorn_iter (int): Number of iterations for Sinkhorn distance computation.
                ii_coeff (float): Coefficient used in computing teacher targets for self-distillation.
                tt_coeff (float): Coefficient used in computing teacher targets for self-distillation.
                remove_diag (bool): Flag to determine whether to remove diagonal elements when computing teacher
                    targets.
            """
        super().__init__()

        if use_warmup != "cosine" and use_warmup != "linear":
            raise ValueError(f"{use_warmup} not supported. Try 'cosine' or 'linear'.")

        # Init main model
        self._student = CLIPModel.from_pretrained(model)

        # If no LR is passed, the default one from the model's configuration will be used
        # Note: Only Visual Transformers are currently supported
        if lr is None:
            model_size = model.split("patch", 1)[1]
            model_name = f"ViT-B/{model_size}"

            with open(f"{CONFIG_DIR}/{VIT_CONFIG_FILE}") as stream:
                config = yaml.safe_load(stream)[model_name]

            self._lr = float(config["learning_rate"])
        else:
            self._lr = lr

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

        # Init self-distillation loss and teacher model
        self._dist_loss = DistillationLoss()
        self._teacher = copy.deepcopy(self._student)
        self._ema_model = ExponentialMovingAverage(self._student.parameters(), decay=ema_decay)

        self._sinkhorn_lambda = sinkhorn_lambda
        self._sinkhorn_iter = sinkhorn_iter
        self._ii_coeff = ii_coeff
        self._tt_coeff = tt_coeff
        self._remove_diag = float(remove_diag)

        # Save hyperparameters when checkpointing
        self.save_hyperparameters()

    def get_embeddings(self, images, text, teacher: bool = False):
        """
        Get embeddings for images and text.

        Args:
            images: The input images.
            text: The input text.
            teacher (bool): Whether to use teacher model for embeddings.

        Returns:
            tuple: A tuple containing image and text embeddings.
        """
        images_embeds = self.encode_image(images=images, teacher=teacher)
        text_embeds = self.encode_text(text=text, teacher=teacher)

        # Normalize embeddings
        images_embeds = images_embeds / images_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

        return images_embeds, text_embeds

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        images, text, raw_text = batch[IMAGE_FIELD], batch[CAPTION_FIELD], batch.pop(RAW_CAPTION_FIELD)

        # Update the teacher model
        self.update_teacher()

        # Get student embeddings and compute unscaled logits
        student_images_embeds, student_text_embeds = self.get_embeddings(images=images, text=text)
        unscaled_student_images_logits = torch.matmul(student_images_embeds, student_text_embeds.t())

        # Compute teacher embeddings and self-distillation targets
        with torch.no_grad():
            teacher_images_embeds, teacher_text_embeds = self.get_embeddings(images=images, text=text, teacher=True)
            teacher_images_embeds = teacher_images_embeds.detach()
            teacher_text_embeds = teacher_text_embeds.detach()
            images_target_prob, text_target_prob = compute_teacher_targets(teacher_images_embeds, teacher_text_embeds,
                                                                           self._ii_coeff, self._tt_coeff,
                                                                           self._sinkhorn_lambda, self._sinkhorn_iter,
                                                                           self._remove_diag)
        # Compute self-distillation loss
        images_dist_loss = self._dist_loss(pred=unscaled_student_images_logits, target_prob=images_target_prob)
        text_dist_loss = self._dist_loss(pred=unscaled_student_images_logits.t(), target_prob=text_target_prob)
        distillation_loss = (images_dist_loss + text_dist_loss) / 2

        # Compute contrastive loss
        contrastive_loss = self.forward(batch).loss

        # Total loss
        loss = (self._alpha * contrastive_loss) + ((1 - self._alpha) * distillation_loss)

        # Compute training metrics
        # CLIP/Sentence-BERT MSE in embeddings similarities
        mse = compute_mse(clip_image_embeddings=student_images_embeds, clip_text_embeddings=student_text_embeds,
                          st_embeddings=self._st_model.encode(raw_text), device=images.device)

        # Compute accuracy (contrastive loss ground truth = identity matrix)
        accuracy = compute_accuracy(images_logits=unscaled_student_images_logits, batch_size=len(images))

        # Log metrics
        self.log_dict({'loss': loss.item(),
                       'mse_sentence_transformer': mse,
                       'acc': accuracy},
                      prog_bar=True, on_step=True, on_epoch=True, logger=True, enable_graph=True)

        return loss

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        images, text, raw_text = batch[IMAGE_FIELD], batch[CAPTION_FIELD], batch.pop(RAW_CAPTION_FIELD)
        images_embeds, text_embeds = self.get_embeddings(images=images, text=text)

        outputs = self.forward(batch)

        accuracy = compute_accuracy(images_logits=outputs.logits_per_image, batch_size=len(images_embeds))
        mse = compute_mse(clip_image_embeddings=images_embeds, clip_text_embeddings=text_embeds,
                          st_embeddings=self._st_model.encode(raw_text), device=images.device)

        self.log_dict({'val_loss': outputs.loss.item(),
                       'val_mse_sentence_transformer': mse,
                       'val_acc': accuracy},
                      prog_bar=True, on_step=True, on_epoch=True, logger=True, enable_graph=True)

        return outputs.loss

    def encode_image(self, images: torch.Tensor, teacher: bool = False) -> torch.FloatTensor:
        """
        Encode images and obtain their embeddings.

        Args:
            images: The input images.
            teacher (bool): Whether to use teacher model for encoding.

        Returns:
            torch.FloatTensor: The image embeddings.
        """
        return self._student.get_image_features(images) if teacher is False else \
            self._teacher.get_image_features(images)

    def encode_text(self, text: torch.Tensor, teacher: bool = False) -> torch.FloatTensor:
        """
        Encode text and obtain their embeddings.

        Args:
            text: The input text.
            teacher (bool): Whether to use teacher model for encoding.

        Returns:
            torch.FloatTensor: The text embeddings.
        """
        return self._student.get_text_features(text) if teacher is False else \
            self._teacher.get_text_features(text)

    def forward(self, inputs, return_loss: bool = True):
        return self._student(**inputs, return_loss=return_loss)

    def update_teacher(self):
        """
        Update the teacher model using Exponential Moving Average (EMA).
        """
        self._ema_model.update(self._student.parameters())
        self._ema_model.copy_to(self._teacher.parameters())

    def configure_optimizers(self) -> dict:
        optimizer = torch.optim.AdamW(params=self._student.parameters(), lr=self._lr, eps=self._eps,
                                      betas=self._betas, weight_decay=self._weight_decay)

        if self._use_warmup == "cosine":
            # Source: https://github.com/PyTorchLightning/pytorch-lightning/issues/5449
            # Source: https://github.com/openai/CLIP/issues/107
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

    @property
    def lr(self) -> float:
        """
        Get the learning rate.

        Returns:
            float: The current learning rate.

        Note:
            This is necessary in order to use PytorchLightning's Tuner.
        """
        return self._lr

    @lr.setter
    def lr(self, lr: float):
        """
        Set the learning rate.

        Args:
            lr (float): The new learning rate to set.

        Note:
            This is necessary in order to use PytorchLightning's Tuner.
        """
        self._lr = lr
