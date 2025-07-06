import os
from typing import Tuple

import torch
import torch.nn as nn
from config.diffconfig import DiffusionConfig, get_model_conf
from diffusion import create_gaussian_diffusion, ddim_steps, make_beta_schedule
from tensorfn import load_config as DiffConfig

from ..base import BaseModelInterface
from .custom_rn import resnet18

cur_dir = os.path.dirname(os.path.abspath(__file__))


class DiffFAS(BaseModelInterface):
    def __init__(
        self,
        modality: str = "rgb",
    ):
        super().__init__()
        self._build_model()
        self.modality = modality

    def _build_model(self):
        conf = get_model_conf()
        conf.in_channels = 3 + 3
        self.model = conf.make_model()
        conf.in_channels = 3 + 3
        self.ema = conf.make_model()
        self.diffconf = DiffConfig(
            DiffusionConfig,
            cur_dir + "config/diffusion.conf",
        )
        self.betas = self.diffconf.diffusion.beta_schedule.make()
        self.diffusion = create_gaussian_diffusion(self.betas, predict_xstart=False)
        self.encoder = self.model.encoder().eval()

    def forward(self, xs, norm_flag: bool = True):
        samples = self.diffusion.p_sample_loop(
            self.model,
            self.encoder,
            x_cond=[val_pose_shuffled, val_img],
            progress=True,
            cond_scale=2,
            sample_initial_noise=250,
            means_size=5,
            var_size=3,
            use_pair=True,
        )
        return cls_outs, feats

    def _prepare_blobs(self, batch: dict):
        xs = batch["img"]
        if self.modality == "mm":
            depths = batch["depth"]
            irs = batch["ir"]
            xs = torch.concat((xs, depths, irs), dim=1)
        return xs

    def forward_train(self, batch: dict):
        blobs = self._prepare_blobs(batch)
        labels = batch["label"]
        logits, _ = self(blobs)
        loss = self.ce_loss(logits, labels)
        return {"loss": loss}

    @torch.no_grad()
    def forward_test(self, batch: dict):
        blobs = self._prepare_blobs(batch)
        logits, _ = self(blobs)
        scores = torch.softmax(logits, dim=-1)[..., 1]
        return {
            "scores": scores,
            "logits": logits,
        }

    @torch.no_grad()
    def show_detail(self, *args, **kwargs):
        return
