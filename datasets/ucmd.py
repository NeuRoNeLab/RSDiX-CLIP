from .captioningDataset import CaptioningDatasetDataModule


class UCMD(CaptioningDatasetDataModule):
    """ Universidad Complutense de Madrid Dataset. """

    def __init__(self, annotations_file: str, img_dir: str, img_transform=None, target_transform=None,
                 batch_size: int = 32, num_workers: int = 0, shuffle: bool = False, custom_tokenizer=None):
        super().__init__(annotations_file, img_dir, img_transform, target_transform, batch_size, num_workers, shuffle,
                         custom_tokenizer)
