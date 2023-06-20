from .captioningDataset import CaptioningDataModule
from utils import TRAIN_SPLIT_PERCENTAGE, VAL_SPLIT_PERCENTAGE


class RSICD(CaptioningDataModule):
    """ Remote Sensing Image Captioning Dataset. """

    def __init__(self, annotations_file: str, img_dir: str, img_transform=None, target_transform=None,
                 train_split_percentage: float = TRAIN_SPLIT_PERCENTAGE,
                 val_split_percentage: float = VAL_SPLIT_PERCENTAGE, batch_size: int = 512, num_workers: int = 0,
                 shuffle: bool = False):
        super().__init__(annotations_file=annotations_file, img_dir=img_dir, img_transform=img_transform,
                         target_transform=target_transform, train_split_percentage=train_split_percentage,
                         val_split_percentage=val_split_percentage, batch_size=batch_size, num_workers=num_workers,
                         shuffle=shuffle)
