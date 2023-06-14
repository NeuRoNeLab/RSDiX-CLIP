import os
import torch
import random

import pandas as pd
import lightning as l

from PIL import Image
from torch import cuda
from torch.backends import mps
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms as t
from torchvision.io import read_image

from utils import DEFAULT_TRANSFORMS, IMAGE_DEFAULT_C, IMAGE_DEFAULT_H, IMAGE_DEFAULT_W
from transformations import BackTranslation


class CaptioningDataset(Dataset):
    # needs to be static in order to track every call to every api
    _BackTranslation = BackTranslation()

    def __init__(self, annotations_file: str, img_dir: str, img_transform=None, target_transform=None,
                 train: bool = False):
        """
            Args:
                annotations_file (string): Path to the file containing the annotations.
                img_dir (string): Directory with all the NAIS_images.
                img_transform (callable, optional): Optional transform to be applied on an image. If None, random
                    transformations will be applied.
                target_transform (callable, optional): Optional transform to be applied on a caption.
                train (bool): Whether to apply transforms to augment data based on the current stage.
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
        self._img_transform = img_transform if img_transform is not None else DEFAULT_TRANSFORMS
        self._target_transform = target_transform
        self._device = "cuda" if cuda.is_available() else "mps" if mps.is_available() else "cpu"
        self._train = train

    @property
    def img_captions(self):
        return self._img_captions

    @property
    def get_img_dir(self) -> str:
        return self._img_dir

    @property
    def img_transform(self):
        return self._img_transform

    @property
    def target_transform(self):
        return self._target_transform

    @property
    def device(self) -> str:
        return self._device

    @property
    def train(self) -> bool:
        return self._train

    def __len__(self) -> int:
        return len(self._img_captions)

    def __getitem__(self, idx):
        """
            Returns a tuple containing the image and the caption.
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
        caption = random.choice(row["sentences"])["raw"]

        # a tensor image shape could not be equal to the shape expected
        if list(image.shape) != [IMAGE_DEFAULT_C, IMAGE_DEFAULT_H, IMAGE_DEFAULT_W]:
            image = t.Resize((IMAGE_DEFAULT_H, IMAGE_DEFAULT_W), antialias=True)(image)

        if self._train:
            image = self._img_transform(image)
            # back translation
            caption = self._BackTranslation(caption)

            if self._target_transform:
                caption = self._target_transform(caption)

        return image, caption


class CaptioningDataModule(l.LightningDataModule):
    def __init__(self, annotations_file: str, img_dir: str, img_transform=None, target_transform=None,
                 train_split_percentage: float = 80, val_split_percentage: float = 10, batch_size: int = 32,
                 num_workers: int = 0, shuffle: bool = False):
        """
            Args:
                annotations_file (string): Path to the file containing the annotations.
                img_dir (string): Directory with all the NAIS_images.
                img_transform (callable, optional): Optional transform to be applied on an image in order to perform data
                    augmentation. If None, random transformations will be applied.
                target_transform (callable, optional): Optional transform to be applied on a caption.
                train_split_percentage (float): The training set split percentage. If smaller than 100, the remaining
                    will be divided between the validation and test set.
                val_split_percentage (float): The validation set split percentage. If train_split + val_split is smaller
                    than 100, the remaining will be used to split train set.
                batch_size (int): The batch size of each dataloader.
                num_workers (int, optional): The number of workers in the DataLoader. Defaults to 0.
                shuffle (bool, optional): Whether to have shuffling behavior during sampling. Defaults to False.
            """
        super().__init__()

        self._annotations_file = annotations_file
        self._img_dir = img_dir
        self._img_transform = img_transform
        self._target_transform = target_transform
        self._train_split_percentage = train_split_percentage
        self._val_split_percentage = val_split_percentage
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._shuffle = shuffle

        self._train_set = None
        self._val_set = None
        self._test_set = None

    def setup(self, stage: str):
        train = stage == 'fit'
        dataset = CaptioningDataset(annotations_file=self._annotations_file, img_dir=self._img_dir,
                                    img_transform=self._img_transform, target_transform=self._target_transform,
                                    train=train)

        train_split = int(len(dataset) * self._train_split_percentage / 100)
        remaining_split = len(dataset) - train_split
        val_split = remaining_split - int(len(dataset) * self._val_split_percentage / 100)
        test_split = remaining_split - val_split

        self._train_set, self._val_set, self._test_set = random_split(dataset, [train_split, val_split, test_split])

    def train_dataloader(self):
        return DataLoader(self._train_set, batch_size=self._batch_size, num_workers=self._num_workers,
                          drop_last=True, shuffle=self._shuffle)

    def val_dataloader(self):
        return DataLoader(self._val_set, batch_size=self._batch_size, num_workers=self._num_workers,
                          drop_last=True)

    def test_dataloader(self):
        return DataLoader(self._test_set, batch_size=self._batch_size, num_workers=self._num_workers,
                          drop_last=True)
