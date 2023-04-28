from .captioningDataset import CaptioningDataset


class UCMD(CaptioningDataset):
    """ Universidad Complutense de Madrid Dataset. """

    def __init__(self, annotations_file, img_dir, img_transform=None, target_transform=None):
        super().__init__(annotations_file, img_dir, img_transform, target_transform)
