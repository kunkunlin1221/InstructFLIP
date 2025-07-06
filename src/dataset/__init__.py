import random
from typing import Any, List

from torch.utils.data import ConcatDataset, Dataset

from .ca_dataset import CADataset
from .instruct import (AblationCAInstructDataset,
                       CAInstructDataset, CAMCDataset, RGBMCDataset,
                       RGBInstructDataset)
from .rgb_dataset import RGBDataset
from .utils import RestoreImageMixin


class BalanceDataset(RestoreImageMixin, Dataset):
    color_channel = "rgb"
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    def __init__(
        self,
        datasets_kwargs: List[dict],
        dataset_size: int = 80000,
    ):
        self.datasets = [build_dataset(**kwargs) for kwargs in datasets_kwargs]
        self.dataset_size = dataset_size

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, ind) -> Any:
        dataset = self.datasets[ind % len(self.datasets)]
        return dataset[random.randint(0, len(dataset) - 1)]


class ConcatRGBDataset(RestoreImageMixin, Dataset):
    color_channel = "rgb"
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    def __init__(self, datasets_kwargs: List[dict]):
        self.dataset = ConcatDataset(
            [build_dataset(**kwargs) for kwargs in datasets_kwargs])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, ind) -> Any:
        return self.dataset[ind]


BASE_DATASETs = {
    'RGBDataset': RGBDataset,
    'CADataset': CADataset,
    'BalanceDataset': BalanceDataset,
    'ConcatRGBDataset': ConcatRGBDataset,
    'CAInstructDataset': CAInstructDataset,
    'RGBInstructDataset': RGBInstructDataset,
    'CAInstructDataset': CAInstructDataset,
    'AblationCAInstructDataset': AblationCAInstructDataset,
    'CAMCDataset': CAMCDataset,
    'RGBMCDataset': RGBMCDataset,
}


def build_dataset(cls_name, kwargs):
    dataset_cls = BASE_DATASETs.get(cls_name)
    dataset = dataset_cls(**kwargs)
    return dataset
