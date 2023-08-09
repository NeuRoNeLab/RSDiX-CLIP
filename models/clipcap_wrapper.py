import lightning as l

from typing import Optional, Union

from clip import CLIPWrapper
from models.clipcap import ClipCaptionModel, generate_caption
from models.clipcap import MappingType
from torch.nn import functional as f

from utils import IMAGE_FIELD, BETAS, GPT2_CAPTION_TOKENS_FIELD, MASK_FIELD, METRICS, METEOR, BLEU, \
    MIN_BLEU, MAX_BLEU, ALLOWED_METRICS, RAW_CAPTION_FIELD


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
                 checkpoint_path: str = None,
                 metrics: Union[str, list] = ALLOWED_METRICS,
                 use_beam_search: bool = False,
                 every_n_batches: int = 10):
        super().__init__()

        if isinstance(metrics, str):
            metrics = [metrics]

        for _ in metrics:
            if _ not in ALLOWED_METRICS:
                raise Exception(f"metric `{_} not allowed. ALLOWED METRICS: f{ALLOWED_METRICS}")

        if load_from_checkpoint:
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
        self._metrics = metrics
        self._use_beam_search = use_beam_search
        self._every_n_batches = every_n_batches

    def _compute_loss(self, tokens, prefix, mask):
        outputs = self._clipcap(tokens, prefix, mask)
        logits = outputs.logits[:, self._clipcap.prefix_length - 1: -1]
        loss = f.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens.flatten(), ignore_index=0)

        return loss

    def training_step(self, batch, batch_idx):
        images, tokens, mask = batch[IMAGE_FIELD], batch[GPT2_CAPTION_TOKENS_FIELD], batch[MASK_FIELD]

        # get CLIP images embeddings that will be used as the prefix by the captioning model
        images_embeds = self._clip_wrapper.encode_image(images)
        loss = self._compute_loss(tokens, images_embeds, mask)

        self.log_dict({'loss': loss.item()}, prog_bar=True, on_step=True, on_epoch=True, logger=True,
                      enable_graph=True)

        return loss

    def validation_step(self, batch, batch_idx):
        images, tokens, mask, raw_captions = batch[IMAGE_FIELD], batch[GPT2_CAPTION_TOKENS_FIELD], batch[MASK_FIELD], \
            batch[RAW_CAPTION_FIELD]

        # get CLIP images embeddings that will be used as the prefix by the captioning model
        images_embeds = self._clip_wrapper.encode_image(images)
        val_loss = self._compute_loss(tokens, images_embeds, mask)

        metrics = {metric: 0.0 for metric in self._metrics}
        # temporary
        preds = [generate_caption(img=images,
                                  clip_encoder=self._clip_wrapper,
                                  tokenizer=self._clipcap.gpt,
                                  model=self._clipcap,
                                  use_beam_search=self._use_beam_search)]
        
        for metric in self._metrics:
            if metric == METEOR:
                try:
                    value, _ = METRICS[metric](candidates=preds, mult_references=raw_captions)
                    metrics[metric] = value[metric].item()
                except ValueError as e:
                    print(f"Meteor could not be computed due to error {e.with_traceback(None)} "
                          f"on the couple: ({preds}, {raw_captions}). ")
            elif metric == BLEU:
                for j in range(MIN_BLEU, MAX_BLEU + 1):
                    blue_j = f"{BLEU}{j}"
                    if blue_j in self._metrics:
                        value, _ = METRICS[metric](candidates=preds, mult_references=raw_captions, n=j)
                        metrics[blue_j] = value[blue_j].item()
            else:
                value, _ = METRICS[metric](candidates=preds, mult_references=raw_captions)
                metrics[metric] = value[metric].item()

        self.log_dict({'val_loss': val_loss.item(), 'metrics': metrics}, prog_bar=True, on_step=True,
                      on_epoch=True, logger=True, enable_graph=True)

        return val_loss
