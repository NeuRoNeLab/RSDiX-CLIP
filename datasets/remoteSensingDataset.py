import pandas as pd
import torch
from torch.utils.data import Dataset

from .constants import DEFAULT_TRANSFORMS


class RemoteSensingDataset(Dataset):
    """"""

    def __init__(self, annotations_file, img_dir, img_transform=None, target_transform=None):
        """
            Arguments:
                annotations_file (string): Path to the file containing the annotations.
                img_dir (string): Directory with all the images.
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
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def __len__(self) -> int:
        return len(self._img_captions)
