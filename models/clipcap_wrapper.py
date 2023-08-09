from typing import Optional

import lightning as l

from clip import CLIPWrapper
from clipcap import ClipCaptionModel
from models.clipcap import MappingType
from torch.nn import functional as f

from utils import IMAGE_FIELD, CAPTION_FIELD, BETAS


class CLIPCapWrapper(l.LightningModule):

    def __init__(self,
                 prefix_length: int,
                 clip_length: Optional[int] = None,
                 prefix_size: int = 512,
                 num_layers: int = 8,
                 mapping_type: MappingType = MappingType.MLP,
                 dropout_transformer: float = 0.0,
                 dropout_gpt2: Optional[float] = None,
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
                 checkpoint_path: str = None):
        super().__init__()

        if load_from_checkpoint:
            if checkpoint_path is None:
                raise Exception("`checkpoint_path` can not be None when `load_from_checkpoint` is True")

            self._clip_wrapper = CLIPWrapper.load_from_checkpoint(checkpoint_path=checkpoint_path)
        else:
            self._clip_wrapper = CLIPWrapper(model=model, lr=lr, alpha=alpha, ema_decay=ema_decay,
                                             weight_decay=weight_decay, start_factor=start_factor,
                                             end_factor=end_factor, total_iters=total_iters,
                                             use_warmup=use_warmup, warmup_steps=warmup_steps, eps=eps, betas=betas,
                                             sinkhorn_lambda=sinkhorn_lambda, sinkhorn_iter=sinkhorn_iter,
                                             ii_coeff=ii_coeff, tt_coeff=tt_coeff, remove_diag=remove_diag)

        self._clipcap = ClipCaptionModel(prefix_length=prefix_length, clip_length=clip_length, prefix_size=prefix_size,
                                         num_layers=num_layers, mapping_type=mapping_type,
                                         dropout_transformer=dropout_transformer, dropout_gpt2=dropout_gpt2)

    def training_step(self, batch, batch_idx):
        images, captions = batch[IMAGE_FIELD], batch[CAPTION_FIELD]

        # get CLIP images embeddings that will be used as the prefix by the captioning model
        images_embeds = self._clip_wrapper.encode_image(images)
        tokens, mask = 1, 1
        outputs = self._clipcap(tokens, images_embeds, mask)
        logits = outputs.logits[:, self._clipcap.prefix_length - 1: -1]
        loss = f.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens.flatten(), ignore_index=0)

        self.log_dict({'loss': loss.item()}, prog_bar=True, on_step=True, on_epoch=True, logger=True,
                      enable_graph=True)

        return loss

    def validation_step(self, batch, batch_idx):
        images, captions = batch[IMAGE_FIELD], batch[CAPTION_FIELD]

        # get CLIP images embeddings that will be used as the prefix by the captioning model
        images_embeds = self._clip_wrapper.encode_image(images)
        tokens, mask = 1, 1
        outputs = self._clipcap(tokens, images_embeds, mask)
        logits = outputs.logits[:, self._clipcap.prefix_length - 1: -1]
        loss = f.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens.flatten(), ignore_index=0)

        self.log_dict({'val_loss': loss.item()}, prog_bar=True, on_step=True, on_epoch=True, logger=True,
                      enable_graph=True)

        return loss
