from .captioningDataset import CaptioningDataset


class RSICD(CaptioningDataset):
    """ Remote Sensing Image Captioning Dataset. """

    def __init__(self, annotations_file: str, img_dir: str, img_transform=None, target_transform=None):
        super().__init__(annotations_file, img_dir, img_transform, target_transform)
