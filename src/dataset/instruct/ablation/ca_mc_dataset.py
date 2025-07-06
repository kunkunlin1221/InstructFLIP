# multi class dataset
from pathlib import Path
from typing import Any, Optional

import mmcv
import pandas as pd
import torch
from PIL import Image
import numpy as np
from torch.utils.data import Dataset

from ...transforms import (get_ssl_transforms, get_train_transforms,
                          get_valid_transforms)
from ...utils import RestoreImageMixin
from .._preprocess import MEAN, STD


class CAMCDataset(RestoreImageMixin, Dataset):
    color_channel = "rgb"
    mean = torch.tensor(MEAN)
    std = torch.tensor(STD)

    def __init__(
        self,
        csv_file: str,
        train: bool = False,
        mode: str = 'all',
        ssl_mode: bool = False,
        name: str = None,
        num_samples: int = None,
        resolution: Optional[str] = None,
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

        if resolution == 'low':
            self.data = data[data['quality'] == 0]
        elif resolution == 'mid':
            self.data = data[data['quality'] == 1]
        elif resolution == 'high':
            self.data = data[data['quality'] == 2]

        self.mother_folder = csv_file.parent
        if num_samples is not None:
            data = data.iloc[:num_samples]
        self.data = data
        if train:
            self.transforms = get_train_transforms(MEAN, STD)
        else:
            self.transforms = get_valid_transforms(MEAN, STD)
        self.ssl_transforms = get_ssl_transforms(MEAN, STD)

        self.train = train
        self.ssl_mode = ssl_mode
        self.name = name


    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, ind) -> Any:
        img_path, spoof_type, illumination, envir, is_fake, quality = self.data.iloc[ind].tolist()
        img_path = self.mother_folder / img_path
        img_path = img_path.with_suffix('.jpg')
        img = mmcv.imread(img_path, channel_order=self.color_channel).astype('float32')
        img = Image.fromarray(img.astype('uint8'))

        out = {}
        out["img"] = self.transforms(img)
        out["label"] = is_fake
        out["img_path"] = str(img_path)

        out['content_input'] = ""
        out['content_label'] = spoof_type

        i = np.random.randint(0, 3)

        out['style_input'] = ""
        if i == 0:
            out['style_label'] = illumination
        elif i == 1:
            out['style_label'] = envir + 5
        else:
            out['style_label'] = quality + 8

        if self.ssl_mode:
            out["ssl_img1"] = self.ssl_transforms(img)
            out["ssl_img2"] = self.ssl_transforms(img)

        out["mean"] = self.mean
        out['std'] = self.std
        out['video_id'] = ind

        return out
