import os
from typing import Optional, Union

import lightning as l
import torch
from transformers import GPT2Tokenizer, get_linear_schedule_with_warmup

from models.clip import CLIPWrapper
from models.clipcap import ClipCaptionModel, generate_caption
from models.clipcap import MappingType
from models.clipcap.model_utils import compute_loss
from utils import IMAGE_FIELD, BETAS, GPT2_CAPTION_TOKENS_FIELD, ALLOWED_METRICS, RAW_CAPTION_FIELD, GPT2_MASK_FIELD, \
    METEOR, METRICS, BLEU, MIN_BLEU, MAX_BLEU


class CLIPCapWrapper(l.LightningModule):

    def __init__(self,
                 prefix_length: int,
                 clip_length: Optional[int] = None,
                 prefix_size: int = 512,
                 num_layers: int = 8,
                 mapping_type: MappingType = MappingType.MLP,
                 dropout_transformer: float = 0.0,
                 dropout_gpt2: Optional[float] = None,
                 clipcap_lr: float = 1e-3,
                 clipcap_weight_decay: float = 0.1,
                 clipcap_warmup_steps: int = 5000,
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
                 load_from_checkpoint: bool = False,
                 checkpoint_path: str = None,
                 metrics: Union[str, list] = ALLOWED_METRICS,
                 use_beam_search: bool = False,
                 tokenizer: str = "gpt2",
                 pad_token: str = None,
                 every_n_batches: int = 10,
                 freeze_clip_encoder: bool = True):
        super().__init__()

        if isinstance(metrics, str):
            metrics = [metrics]

        for _ in metrics:
            if _ not in ALLOWED_METRICS:
                raise Exception(f"metric `{_} not allowed. ALLOWED METRICS: f{ALLOWED_METRICS}")

        if load_from_checkpoint:
            self._clip_encoder = CLIPWrapper.load_from_checkpoint(checkpoint_path=checkpoint_path)
        else:
            self._clip_encoder = CLIPWrapper(model=model, lr=lr, alpha=alpha, ema_decay=ema_decay,
                                             weight_decay=weight_decay, start_factor=start_factor,
                                             end_factor=end_factor, total_iters=total_iters,
                                             use_warmup=use_warmup, warmup_steps=warmup_steps, eps=eps, betas=betas,
                                             sinkhorn_lambda=sinkhorn_lambda, sinkhorn_iter=sinkhorn_iter,
                                             ii_coeff=ii_coeff, tt_coeff=tt_coeff, remove_diag=remove_diag)

        if freeze_clip_encoder:
            self._clip_encoder.freeze()

        self._clipcap = ClipCaptionModel(prefix_length=prefix_length, clip_length=clip_length, prefix_size=prefix_size,
                                         num_layers=num_layers, mapping_type=mapping_type,
                                         dropout_transformer=dropout_transformer, dropout_gpt2=dropout_gpt2)
        self._clipcap_lr = clipcap_lr
        self._clipcap_weight_decay = clipcap_weight_decay
        self._clipcap_warmup_steps = clipcap_warmup_steps
        self._metrics = metrics
        self._use_beam_search = use_beam_search
        self._every_n_batches = every_n_batches
        self._gpt2_tokenizer = GPT2Tokenizer.from_pretrained(tokenizer)

        if BLEU in metrics:
            for j in range(MIN_BLEU, MAX_BLEU + 1):
                metrics.append(f"{BLEU}{j}")
            metrics.remove(BLEU)

        self._avg_metrics = {metric: 0.0 for metric in self._metrics}
        self._avg_metrics_idx = 0

        if METEOR in self._metrics:
            self._no_meteor_count = 0

        if pad_token is not None:
            self._gpt2_tokenizer.pad_token = pad_token
        elif self._gpt2_tokenizer.pad_token is None:
            self._gpt2_tokenizer.pad_token = self._gpt2_tokenizer.eos_token

        self.save_hyperparameters()

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        images, tokens, mask = batch[IMAGE_FIELD], batch[GPT2_CAPTION_TOKENS_FIELD], batch[GPT2_MASK_FIELD]

        # get CLIP images embeddings that will be used as the prefix by the captioning model
        prefix = self._clip_encoder.encode_image(images)
        # normalize
        prefix /= prefix.norm(p=2, dim=-1, keepdim=True)
        loss = compute_loss(self._clipcap, tokens, prefix, mask)

        self.log_dict({'loss': loss.item()}, prog_bar=True, on_step=True, on_epoch=True, logger=True,
                      enable_graph=True)

        return loss

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        images, tokens, mask, raw_captions = (batch[IMAGE_FIELD], batch[GPT2_CAPTION_TOKENS_FIELD],
                                              batch[GPT2_MASK_FIELD], batch[RAW_CAPTION_FIELD])

        raw_captions = [[rc] for rc in raw_captions]
        # get CLIP images embeddings that will be used as the prefix by the captioning model
        prefix = self._clip_encoder.encode_image(images)
        # normalize
        prefix /= prefix.norm(p=2, dim=-1, keepdim=True)
        val_loss = compute_loss(self._clipcap, tokens, prefix, mask)

        if self.trainer.sanity_checking is False and batch_idx > 0 and batch_idx % self._every_n_batches == 0:
            preds = generate_caption(imgs=images,
                                     clip_encoder=self._clip_encoder,
                                     tokenizer=self._gpt2_tokenizer,
                                     model=self._clipcap,
                                     use_beam_search=self._use_beam_search)
            for metric in self._metrics:
                if metric == METEOR:
                    try:
                        value, _ = METRICS[metric](candidates=preds, mult_references=raw_captions,
                                                   java_path=os.getenv("JAVA_HOME"))
                        value = value[metric].item()
                        self._avg_metrics[METEOR] = self._avg_metrics[metric] + 1 / (
                                    self._avg_metrics_idx + 1 - self._no_meteor_count) * (
                                                            value - self._avg_metrics[metric])
                    except ValueError as e:
                        print(f"Meteor could not be computed due to error {e.with_traceback(None)} "
                              f"on the couple: ({preds}, {raw_captions}). "
                              f"Increasing the no_meteor_count to {self._no_meteor_count}")
                        self._no_meteor_count += 1
                else:
                    if BLEU in metric:
                        j = int(metric.split("_")[1])
                        value, _ = METRICS[BLEU](candidates=preds, mult_references=raw_captions, n=j)
                    else:
                        value, _ = METRICS[metric](candidates=preds, mult_references=raw_captions)
                    value = value[metric].item()
                    self._avg_metrics[metric] = self._avg_metrics[metric] + 1 / (self._avg_metrics_idx + 1) * (
                                value - self._avg_metrics[metric])
                self._avg_metrics_idx = self._avg_metrics_idx + 1

        self._avg_metrics['val_loss'] = val_loss.item()
        self.log_dict(self._avg_metrics, prog_bar=True, on_step=True,
                      on_epoch=True, logger=True, enable_graph=True)
        return val_loss

    def configure_optimizers(self) -> dict:
        optimizer = torch.optim.AdamW(params=self._clipcap.parameters(), lr=self._clipcap_lr,
                                      weight_decay=self._clipcap_weight_decay)
        lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self._clipcap_warmup_steps,
                                                       num_training_steps=self.trainer.estimated_stepping_batches)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step"
            }
        }

    @property
    def clip_encoder(self):
        return self._clip_encoder

    @property
    def gpt2_tokenizer(self):
        return self._gpt2_tokenizer

    @property
    def clipcap(self):
        return self._clipcap

    @property
    def clipcap_lr(self):
        return self._clipcap_lr

    @clipcap_lr.setter
    def clipcap_lr(self, lr):
        self._clipcap_lr = lr
