import torch
import random
from torchvision import transforms as t

DEFAULT_TRANSFORMS = t.RandomApply(torch.nn.ModuleList(
    [t.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 3.85)),
     t.RandomRotation(degrees=(0, 180)), t.RandomHorizontalFlip(p=0.5), t.RandomVerticalFlip(p=0.5),
     t.RandomAutocontrast(), t.RandomAdjustSharpness(sharpness_factor=random.uniform(0.5, 2)),
     t.ColorJitter(brightness=.5, contrast=(1, 2))]))

IMAGE_FIELD = "image"
CAPTION_FIELD = "caption"
IMAGE_DEFAULT_DIMS = (224, 224)