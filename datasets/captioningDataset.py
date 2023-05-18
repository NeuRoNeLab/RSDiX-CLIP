import os
import random
import json
import xmltodict

import torch
import pandas as pd
import pytorch_lightning as pl

from torchvision.io import read_image
from torchvision import transforms as t
from torch.utils.data import Dataset, DataLoader
from torch.backends import mps
from torch import cuda

from PIL import Image

from .constants import DEFAULT_TRANSFORMS, IMAGE_FIELD, CAPTION_FIELD, \
    IMAGE_DEFAULT_C, IMAGE_DEFAULT_H, IMAGE_DEFAULT_W

from .transformations import BackTranslation


class CaptioningDataset(Dataset):
    """ The class itself is used to gather all common functionalities and operations
        among datasets instances and to standardize how samples are returned. """

    __back_translation = BackTranslation(from_language="en")

    def __init__(self, annotations_file: str, img_dir: str, img_transform=None, target_transform=None):
        """
            Arguments:
                annotations_file (string): Path to the file containing the annotations.
                img_dir (string): Directory with all the NAIS_images.
                img_transform (callable, optional): Optional transform to be applied on an image in order to
                    perform data augmentation. If None, random transformations will be applied.
                target_transform (callable, optional): Optional transform to be applied on a caption.
        """
        # get annotations_file extension
        annotations_file_ext = annotations_file.split(".")[-1]
        if annotations_file_ext == "json":
            self._img_captions = pd.read_json(annotations_file)
        elif annotations_file_ext == "csv":
            self._img_captions = pd.read_csv(annotations_file)
        else:
            raise Exception(f"annotations_file type: '{annotations_file_ext}' not supported. JSON and CSV format "
                            f"only are supported.")

        self._img_dir = img_dir

        if img_transform:
            self._img_transform = img_transform
        else:
            self._img_transform = DEFAULT_TRANSFORMS

        self._target_transform = target_transform
        self._device = ("cuda" if cuda.is_available() else "mps" if mps.is_available() else "cpu")

    def __len__(self) -> int:
        return len(self._img_captions)

    def __getitem__(self, idx) -> dict:
        """
            Returns a dictionary containing the image and the caption.
            Arguments:
                idx (int, Tensor): The index of the item to return.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self._img_captions.iloc[idx, 0]
        img_name = os.path.join(self._img_dir, row['filename'])
        img_ext = img_name.split(".")[-1]

        if img_ext != 'jpeg' and img_name != 'png':
            image = t.PILToTensor()(Image.open(img_name))
        else:
            image = read_image(img_name).to(self._device)
        # get a random sentence from the five sentences associated to each image
        sentence = row["sentences"][random.randint(0, len(row["sentences"]) - 1)]
        caption = sentence["raw"]

        # a tensor image shape could not be equal to the shape expected
        if list(image.shape) != [IMAGE_DEFAULT_C, IMAGE_DEFAULT_H, IMAGE_DEFAULT_W]:
            image = t.Resize((IMAGE_DEFAULT_H, IMAGE_DEFAULT_W), antialias=True)(image)

        image = self._img_transform(image)

        # back translation
        caption = self.__back_translation(caption)

        if self._target_transform:
            caption = self._target_transform(caption)

        return {IMAGE_FIELD: image, CAPTION_FIELD: caption}


class CaptioningDatasetDataModule(pl.LightningDataModule):
    def __init__(self, annotations_file: str, img_dir: str, img_transform=None, target_transform=None,
                 batch_size: int = 32):
        super().__init__()
        self._annotations_file = annotations_file
        self._img_dir = img_dir
        self._img_transform = img_transform
        self._target_transform = target_transform
        self._batch_size = batch_size

    def setup(self, stage: str):
        self.data = CaptioningDataset(self._annotations_file, self._img_dir, self._img_transform,
                                      self._target_transform)

    def train_dataloader(self):
        return DataLoader(self.data, batch_size=self._batch_size)


def nais_to_json(annotations_file: str, json_file_name: str = "dataset_nais"):
    with open(annotations_file) as f:
        data_dict = xmltodict.parse(f.read())

    data_dict = data_dict["annotations"]
    images = {"images": []}

    for image in data_dict["image"]:
        image_data = {"filename": image["@name"], "imgid": int(image["@id"])}
        sentences = []
        for mask in image["mask"]:
            sentences.append({"raw": mask["@label"]})
        image_data["sentences"] = sentences
        images["images"].append(image_data)

    # get annotations_file directory
    data_dir = os.path.dirname(annotations_file)
    print(type(images))
    with open(f"{data_dir}/{json_file_name}.json", "w") as f:
        json.dump(images, f, indent=4)
