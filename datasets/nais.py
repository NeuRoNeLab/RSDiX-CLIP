from .captioningDataset import CaptioningDataModule


class NAIS(CaptioningDataModule):
    """ NAIS Dataset. """

    def __init__(self, annotations_file: str, img_dir: str, img_transform=None, target_transform=None,
                 train_split_percentage: float = 80, val_split_percentage: float = 10, batch_size: int = 32,
                 num_workers: int = 0, shuffle: bool = False):
        super().__init__(annotations_file=annotations_file, img_dir=img_dir, img_transform=img_transform,
                         target_transform=target_transform, train_split_percentage=train_split_percentage,
                         val_split_percentage=val_split_percentage, batch_size=batch_size, num_workers=num_workers,
                         shuffle=shuffle)

