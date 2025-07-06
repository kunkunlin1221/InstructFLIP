from pathlib import Path
from typing import Any

import cv2
import mmcv
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from .transforms import (get_ssl_transforms, get_train_transforms,
                         get_valid_transforms)
from .utils import RestoreImageMixin


class CADataset(RestoreImageMixin, Dataset):
    color_channel = "rgb"
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    def __init__(
        self,
        csv_file: str,
        train: bool = False,
        mode: str = 'all',
        ssl_mode: bool = False,
        name: str = None,
    ):
        super().__init__()
        csv_file = Path(csv_file)
        data = pd.read_csv(csv_file, index_col=False)

        if mode == 'live_only':
            self.data = data[data['is_fake'] == 0]
        elif mode == 'fake_only':
            self.data = data[data['is_fake'] == 1]
        elif mode == 'all':
            pass
        else:
            raise ValueError(f"Unknown mode: {mode}")

        self.mother_folder = csv_file.parent
        self.data = data

        if train:
            self.transforms = get_train_transforms(self.mean, self.std)
        else:
            self.transforms = get_valid_transforms(self.mean, self.std)
        self.ssl_transforms = get_ssl_transforms(self.mean, self.std)

        self.train = train
        self.ssl_mode = ssl_mode
        self.name = name

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, ind) -> Any:
        img_path, *_, is_fake, _ = self.data.iloc[ind].tolist()
        img_path = self.mother_folder / img_path
        img_path = img_path.with_suffix('.jpg')
        img = mmcv.imread(img_path, channel_order=self.color_channel).astype('float32')

        # if self.train:
        #     img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        #     if np.random.randint(2):
        #         img[..., 1] *= np.random.uniform(0.8, 1.2)
        #     img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
        img = Image.fromarray(img.astype('uint8'))

        out = {}
        out["img"] = self.transforms(img)
        out["label"] = is_fake
        out["img_path"] = str(img_path)
        out['video_id'] = ind

        if self.ssl_mode:
            out["ssl_img1"] = self.ssl_transforms(img)
            out["ssl_img2"] = self.ssl_transforms(img)

        out["mean"] = torch.tensor(self.mean)
        out['std'] = torch.tensor(self.std)

        return out

    @classmethod
    def get_option_str(cls):
        return {
            'presentation attack': " ".join([f"({k+1}) {v}" for k, v in cls.spoof_type_label.items()]),
            'illumination condition': " ".join([f"({k+1}) {v}" for k, v in cls.illumination_condition.items()]),
            'enviroment': " ".join([f"({k+1}) {v}" for k, v in cls.environment.items()]),
            'camera quality': " ".join([f"({k+1}) {v}" for k, v in cls.camera_quality.items()]),
        }

    @classmethod
    def get_options(cls):
        return {
            'presentation attack': [f"({k+1}) {v}" for k, v in cls.spoof_type_label.items()],
            'illumination condition': [f"({k+1}) {v}" for k, v in cls.illumination_condition.items()],
            'enviroment': [f"({k+1}) {v}" for k, v in cls.environment.items()],
            'camera quality': [f"({k+1}) {v}" for k, v in cls.camera_quality.items()],
        }
