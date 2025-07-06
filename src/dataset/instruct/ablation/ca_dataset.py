from pathlib import Path
from random import choice
from typing import Any, Optional

import mmcv
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from ...transforms import (get_ssl_transforms, get_train_transforms,
                           get_valid_transforms)
from ...utils import RestoreImageMixin
from .._ca_instruction import QA
from .._preprocess import MEAN, STD


class AblationCAInstructDataset(RestoreImageMixin, Dataset):
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
        category_ab_mode: int = 0,
        num_style_prompt: int = 3,
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

        if category_ab_mode == 1:
            label_list = [0, 3, 4, 8, 10]
        elif category_ab_mode == 2:
            label_list = [0, 1, 3, 4, 6, 7, 8, 10]
        else:
            label_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        data = data[data['spoof_type'].isin(label_list)]
        self.data = data
        if train:
            self.transforms = get_train_transforms(MEAN, STD)
        else:
            self.transforms = get_valid_transforms(MEAN, STD)
        self.ssl_transforms = get_ssl_transforms(MEAN, STD)

        self.train = train
        self.ssl_mode = ssl_mode
        self.name = name

        self.qa_dict = QA
        self.content_keys = ['presentation attack']
        self.style_keys = ['illumination condition', 'enviroment', 'camera quality'][:num_style_prompt]

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

        # generat q and a
        content_key = choice(self.content_keys)
        out['content_input'] = self.qa_dict[content_key]['question']
        out['content_output'] = self.qa_dict[content_key]['options'][spoof_type]

        style_key = choice(self.style_keys) if is_fake else 'camera quality'
        ans = {
            'illumination condition': illumination-1,
            'enviroment': envir-1,
            'camera quality': quality,
        }
        out['style_input'] = self.qa_dict[style_key]['question']
        out['style_output'] = self.qa_dict[style_key]['options'][ans[style_key]]

        out['style_inputs'] = [self.qa_dict[key]['question'] for key in self.style_keys]
        out['style_outputs'] = [self.qa_dict[key]['options'][ans[key]] for key in self.style_keys]

        if self.ssl_mode:
            out["ssl_img1"] = self.ssl_transforms(img)
            out["ssl_img2"] = self.ssl_transforms(img)

        out["mean"] = self.mean
        out['std'] = self.std
        out['video_id'] = ind

        return out
