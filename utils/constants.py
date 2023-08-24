from typing import Final

import torch
from torchvision import transforms as t

from transformations import RandomSharpness
from aac_metrics.functional import bleu, rouge_l, meteor, sbert_sim


IMAGE_DEFAULT_C = 3
IMAGE_DEFAULT_W = 224
IMAGE_DEFAULT_H = 224

RANDOM_ROTATION_DEGREES_MIN = 0
RANDOM_ROTATION_DEGREES_MAX = 180
RANDOM_SHARPNESS_MIN = 0.75
RANDOM_SHARPNESS_MAX = 2
COLOR_JITTER_SATURATION_MIN = 0.65
COLOR_JITTER_SATURATION_MAX = 2.5
COLOR_JITTER_BRIGHTNESS_MIN = 0.5
COLOR_JITTER_BRIGHTNESS_MAX = 1.10
RANDOM_RESIZED_CROP_SCALE_MIN = 0.6
RANDOM_RESIZED_CROP_SCALE_MAX = 1.0
RANDOM_RESIZED_CROP_RATIO_MIN = 1.0
RANDOM_RESIZED_CROP_RATIO_MAX = 1.0

DEFAULT_TRANSFORMS = t.RandomApply(torch.nn.ModuleList(
    [t.RandomRotation(degrees=(RANDOM_ROTATION_DEGREES_MIN, RANDOM_ROTATION_DEGREES_MAX)),
     t.RandomHorizontalFlip(p=1), t.RandomVerticalFlip(p=1),
     RandomSharpness(RANDOM_SHARPNESS_MIN, RANDOM_SHARPNESS_MAX, p=1),
     t.ColorJitter(brightness=(COLOR_JITTER_BRIGHTNESS_MIN, COLOR_JITTER_BRIGHTNESS_MAX),
                   saturation=(COLOR_JITTER_SATURATION_MIN, COLOR_JITTER_BRIGHTNESS_MAX)),
     t.RandomResizedCrop(size=[IMAGE_DEFAULT_W, IMAGE_DEFAULT_H],
                         scale=(RANDOM_RESIZED_CROP_SCALE_MIN, RANDOM_RESIZED_CROP_SCALE_MAX),
                         ratio=(RANDOM_RESIZED_CROP_RATIO_MIN, RANDOM_RESIZED_CROP_RATIO_MAX), antialias=True)]))

TRAIN_SPLIT_PERCENTAGE = 80
VAL_SPLIT_PERCENTAGE = 10

CONFIG_DIR = "models/clip/configs"
VIT_CONFIG_FILE = "ViT.yaml"

BETAS = (0.9, 0.99)
BATCH_SIZE = 512
MINIBATCH_SIZE = 0
IMAGE_FIELD = "pixel_values"
CAPTION_FIELD = "input_ids"
RAW_CAPTION_FIELD = "raw_captions"
GPT2_CAPTION_TOKENS_FIELD = "gpt2_caption_tokens"
GPT2_MASK_FIELD = "gpt2_mask"
METEOR: Final[str] = "meteor"
ROUGE_L: Final[str] = "rouge_l"
SBERT_SIM: Final[str] = "sbert_sim"
BLEU: Final[str] = "bleu_"
MIN_BLEU: Final[int] = 1
MAX_BLEU: Final[int] = 4
ALLOWED_METRICS = [METEOR, SBERT_SIM, ROUGE_L, BLEU, f'{BLEU}1', f'{BLEU}2', f'{BLEU}3', f'{BLEU}4']
METRICS = {
    METEOR: meteor,
    SBERT_SIM: sbert_sim,
    ROUGE_L: rouge_l,
    BLEU: bleu
}
CLIP_MAX_LENGTH = 77
