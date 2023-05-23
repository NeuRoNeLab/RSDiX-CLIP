from .captioningDataset import CaptioningDataModule


class UCMD(CaptioningDataModule):
    """ Universidad Complutense de Madrid Dataset. """

    def __init__(self, annotations_file: str, img_dir: str, img_transform=None, target_transform=None,
                 train_split: float = 80, val_split: float = 10, batch_size: int = 32,
                 num_workers: int = 0, shuffle: bool = False, custom_tokenizer=None):
        super().__init__(annotations_file=annotations_file, img_dir=img_dir, img_transform=img_transform,
                         target_transform=target_transform, train_split=train_split, val_split=val_split,
                         batch_size=batch_size, num_workers=num_workers, shuffle=shuffle,
                         custom_tokenizer=custom_tokenizer)

