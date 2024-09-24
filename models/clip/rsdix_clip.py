import lightning as l
import torch
import yaml
import copy

from transformers import AutoModel, AutoProcessor
from typing import Optional, Dict, Any

from loss import DistillationLoss
from sentence_transformers import SentenceTransformer
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from torch.optim.lr_scheduler import LinearLR

from loss.sig_loss import SigmoidLoss
from utils import CONFIG_DIR, VIT_CONFIG_FILE, IMAGE_FIELD, CAPTION_FIELD, BETAS, \
    RAW_CAPTION_FIELD
from .model_utils import compute_mse, compute_accuracy, compute_teacher_targets
from .ema import ExponentialMovingAverage


class RSDiXClip(l.LightningModule):
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
                 remove_diag: bool = False,
                 checkpoint_path: str = None,
                 use_sentence_bert_as_teacher: bool = False,
                 freeze_sentence_bert: bool = True,
                 sentence_bert_model: str = None,
                 use_sigmoid_loss: bool = False):
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
                checkpoint_path (str): Path to the CLIP model checkpoint.
            """
        super().__init__()

        if use_warmup != "cosine" and use_warmup != "linear":
            raise ValueError(f"{use_warmup} not supported. Try 'cosine' or 'linear'.")

        if use_sentence_bert_as_teacher and sentence_bert_model is None:
            raise ValueError(f"sentence_bert_model cannot be None when using sentence bert as a teacher.")

        # Init main model
        self._student = AutoModel.from_pretrained(model)
        self._model_name = model

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

        self._use_sigmoid_loss = use_sigmoid_loss

        # Init self-distillation loss and teacher model
        if use_sigmoid_loss and "siglip" not in model:
            self._dist_loss = SigmoidLoss()
            self._contrastive_loss = SigmoidLoss()
        elif use_sigmoid_loss and "siglip" in model:
            self._dist_loss = SigmoidLoss()
            self._contrastive_loss = SigmoidLoss()  # TODO: delete
        else:
            self._dist_loss = DistillationLoss()

        self._teacher = copy.deepcopy(self._student)

        if checkpoint_path is not None:
            state_dict = torch.load(checkpoint_path)

            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]

            student_state_dict = {key.replace("_student.", ""): value for key, value in state_dict.items() if
                                  "student" in key}
            teacher_state_dict = {key.replace("_teacher.", ""): value for key, value in state_dict.items() if
                                  "teacher" in key}

            self._student.load_state_dict(student_state_dict)
            self._teacher.load_state_dict(teacher_state_dict)

        if use_sentence_bert_as_teacher and sentence_bert_model is not None:
            self._sbert_model = SentenceTransformer(sentence_bert_model)
            # freeze sentence bert
            if freeze_sentence_bert:
                auto_model = self._sbert_model._first_module().auto_model
                for param in auto_model.parameters():
                    param.requires_grad = False

            sentence_bert_dim_embedding = self._sbert_model.encode("a").shape[0]
            processor = AutoProcessor.from_pretrained(model)
            text_input = processor(text="a", truncation=True, return_tensors="pt", padding="max_length")
            text_emb = self._student.get_text_features(**text_input)
            # linear layer to project SBERT embeddings to match CLIP's dimension
            self._proj_linear_sbert_clip = torch.nn.Linear(sentence_bert_dim_embedding, text_emb.shape[1])
        else:
            self._sbert_model = None

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
            teacher_images_embeds, teacher_text_embeds = self.get_embeddings(images=images,
                                                                             text=text if self._sbert_model is None \
                                                                                else raw_text,
                                                                             teacher=True)
            if self._sbert_model is not None:
                teacher_text_embeds = teacher_text_embeds.to(teacher_images_embeds.dtype)
                teacher_text_embeds = self._proj_linear_sbert_clip(teacher_text_embeds)

            teacher_images_embeds = teacher_images_embeds.detach()
            teacher_text_embeds = teacher_text_embeds.detach()
            images_target_prob, text_target_prob = compute_teacher_targets(teacher_images_embeds, teacher_text_embeds,
                                                                           self._ii_coeff, self._tt_coeff,
                                                                           self._sinkhorn_lambda, self._sinkhorn_iter,
                                                                           self._remove_diag)
        # Compute self-distillation loss
        images_dist_loss = self._dist_loss(unscaled_logits=unscaled_student_images_logits, target=images_target_prob)
        text_dist_loss = self._dist_loss(unscaled_logits=unscaled_student_images_logits.t(), target=text_target_prob)
        distillation_loss = (images_dist_loss + text_dist_loss) / 2

        # Compute contrastive loss
        if self._use_sigmoid_loss and "siglip" not in self._model_name:
            contrastive_loss = self._contrastive_loss(unscaled_logits=unscaled_student_images_logits)
        else:
            contrastive_loss = self.forward(batch).loss

        # Total loss
        loss = (self._alpha * contrastive_loss) + ((1 - self._alpha) * distillation_loss)

        # Compute training metrics
        # CLIP/Sentence-BERT MSE in embeddings similarities
        st_embeddings = self._st_model.encode(raw_text)
        mse = compute_mse(clip_image_embeddings=student_images_embeds, clip_text_embeddings=student_text_embeds,
                          st_embeddings=st_embeddings, device=images.device)

        # Compute accuracy (contrastive loss ground truth = identity matrix)
        accuracy = compute_accuracy(images_logits=unscaled_student_images_logits, batch_size=len(images))

        # Log metrics
        self.log_dict({'loss': loss.item(),
                       'mse_sentence_transformer': mse,
                       'acc': accuracy}, sync_dist=True,
                      prog_bar=True, on_step=True, on_epoch=True, logger=True, enable_graph=True)

        return loss

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        images, text, raw_text = batch[IMAGE_FIELD], batch[CAPTION_FIELD], batch.pop(RAW_CAPTION_FIELD)
        images_embeds, text_embeds = self.get_embeddings(images=images, text=text)

        if self._use_sigmoid_loss and "siglip" not in self._model_name:
            logits_per_image = torch.matmul(images_embeds, text_embeds.t())
            loss_item = self._contrastive_loss(unscaled_logits=logits_per_image)
        else:
            outputs = self.forward(batch)
            logits_per_image = outputs.logits_per_image
            loss_item = outputs.loss.item()

        accuracy = compute_accuracy(images_logits=logits_per_image, batch_size=len(images_embeds))
        st_embeddings = self._st_model.encode(raw_text)
        mse = compute_mse(clip_image_embeddings=images_embeds, clip_text_embeddings=text_embeds,
                          st_embeddings=st_embeddings, device=images.device)

        self.log_dict({'val_loss': loss_item,
                       'val_mse_sentence_transformer': mse,
                       'val_acc': accuracy}, sync_dist=True,
                      prog_bar=True, on_step=True, on_epoch=True, logger=True, enable_graph=True)

        return loss_item

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
        # if sbert_model is not None, use the provided sbert model as the teacher for aligning the text embeddings
        # otherwise, use RSDiX-CLIP as the teacher for both image and text embeddings
        if teacher and self._sbert_model is not None:
            return torch.from_numpy(self._sbert_model.encode(text)).to(torch.float32).to(self._student.device)

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
        optimizer = torch.optim.AdamW(params=self.trainer.model.parameters(), lr=self._lr, eps=self._eps,
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

    # https://github.com/Lightning-AI/pytorch-lightning/issues/17798
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """
        Tentative fix for FSDP checkpointing issue
        """
        if not checkpoint.get("state_dict", None):
            state_dict = self.trainer.model.state_dict()
            checkpoint["state_dict"] = state_dict
        return super().on_save_checkpoint(checkpoint)

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

    @property
    def student(self) -> AutoModel:
        """
        Get the student model.

        Returns:
            CLIPModel: The current student model.
        """
        return self._student

    @property
    def ema_model(self) -> ExponentialMovingAverage:
        """
        Get the EMA.

        Returns:
            ExponentialMovingAverage: The current EMA.
        """
        return self._ema_model

    @property
    def sbert_model(self) -> SentenceTransformer:
        """
        Get the SemtemceBERT model.

        Returns:
            SentenceTransformer: The current SentenceBERT model..
        """
        return self._sbert_model
