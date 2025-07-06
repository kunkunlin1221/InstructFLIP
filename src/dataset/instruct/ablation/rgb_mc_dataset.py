import random
from pathlib import Path
from typing import Any, Dict, Union

import mmcv
import torch
from PIL import Image
from torch.utils.data import Dataset

from ...transforms import (get_ssl_transforms, get_train_transforms,
                          get_valid_transforms)
from ...utils import RestoreImageMixin
from .._preprocess import MEAN, STD


class RGBMCDataset(RestoreImageMixin, Dataset):
    color_channel = "rgb"
    mean = torch.tensor(MEAN)
    std = torch.tensor(STD)

    def __init__(
        self,
        data_folder: Union[str, Path],
        train: bool = False,
        ssl_mode: bool = False,
        mode: str = 'all',
        gen_video_id: bool = False,
        name: str = None,
    ):
        super().__init__()

        files = list(Path(data_folder).rglob("*.*"))

        if train:
            random.shuffle(files)

        labels = [int(x.parent.name == 'fake') for x in files]

        if mode == 'live_only':
            inds = [i for i, l in enumerate(labels) if l == 1]
        elif mode == 'fake_only':
            inds = [i for i, l in enumerate(labels) if l == 0]
        elif mode == 'all':
            inds = list(range(len(files)))
        else:
            raise ValueError(f"Unknown mode: {mode}")

        self.files = [files[i] for i in inds]
        self.labels = [labels[i] for i in inds]

        if gen_video_id:
            video_ids = [x.stem.split('_frame')[0] for x in files]
            video_ids_set = list(set(video_ids))
            video_ids = [video_ids_set.index(x) for x in video_ids]
        else:
            video_ids = [0] * len(files)

        self.video_ids = video_ids

        if train:
            self.transforms = get_train_transforms(MEAN, STD)
        else:
            self.transforms = get_valid_transforms(MEAN, STD)
        self.ssl_transforms = get_ssl_transforms(MEAN, STD)

        self.train = train
        self.ssl_mode = ssl_mode
        self.name = name

    def __len__(self):
        return len(self.files)

    def __getitem__(self, ind) -> Dict[str, Any]:
        img_path = self.files[ind]
        img = mmcv.imread(img_path, channel_order=self.color_channel).astype('float32')
        img = Image.fromarray(img.astype('uint8'))

        out = {}
        out["img"] = self.transforms(img)
        out["label"] = self.labels[ind]
        out["img_path"] = str(img_path)
        out["video_id"] = self.video_ids[ind]

        # texts
        out['content_label'] = -1
        out['style_label'] = -1

        if self.ssl_mode:
            out["ssl_img1"] = self.ssl_transforms(img)
            out["ssl_img2"] = self.ssl_transforms(img)

        out["mean"] = self.mean
        out['std'] = self.std

        return out
