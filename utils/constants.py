import torch

from torchvision import transforms as t
from transformations import RandomSharpness


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
                         ratio=(RANDOM_RESIZED_CROP_RATIO_MIN, RANDOM_RESIZED_CROP_RATIO_MAX))]))

TRAIN_SPLIT_PERCENTAGE = 80
VAL_SPLIT_PERCENTAGE = 10
