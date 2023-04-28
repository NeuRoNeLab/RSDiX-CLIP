import os
import random

import torch
from torchvision.io import read_image

from .constants import IMAGE_FIELD, CAPTION_FIELD
from .remoteSensingDataset import RemoteSensingDataset


class RSICD(RemoteSensingDataset):
    """ Remote Sensing Image Captioning Dataset. """

    def __init__(self, annotations_file: str, img_dir: str, img_transform=None, target_transform=None):
        super().__init__(annotations_file, img_dir, img_transform, target_transform)

    def __getitem__(self, idx):
        """
            Returns a dictionary containing the image and the caption.
            Arguments:
                idx (int, Tensor): The index of the item to return.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self._img_captions.iloc[idx, 0]
        img_name = os.path.join(self._img_dir, row['filename'])
        image = read_image(img_name).to(self._device)
        # get a random sentence from the five sentences associated to each image
        sentence = row["sentences"][random.randint(0, len(row["sentences"]) - 1)]
        caption = sentence["raw"]

        image = self._img_transform(image)

        if self._target_transform:
            caption = self._target_transform(caption)

        return {IMAGE_FIELD: image, CAPTION_FIELD: caption}
