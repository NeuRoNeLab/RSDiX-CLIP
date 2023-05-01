from typing import Iterable

from torch.utils.data import ConcatDataset

from .captioningDataset import CaptioningDataset


class ConcatCaptioningDataset(ConcatDataset):

    def __init__(self, datasets: Iterable[CaptioningDataset]):
        super().__init__(datasets)
