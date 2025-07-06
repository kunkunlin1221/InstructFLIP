# from typing import Tuple

import torch
import torch.nn as nn

from ..base import BaseModelInterface

# from .ad_loss import Fake_AdLoss, Real_AdLoss
from .dgfas import DG, Discriminator


class SSDG(BaseModelInterface):
    def __init__(
        self,
        backbon_name: str,
        in_channels: int,
        modality: str = "rgb",
    ):
        super().__init__()
        self._build_model(backbon_name, in_channels=in_channels)
        self.ce_loss = nn.CrossEntropyLoss()
        self.modality = modality

    def _build_model(self, backbon_name, in_channels):
        self.net = DG(backbon_name, in_channels=in_channels)
        self.ad_net_real = Discriminator()
        self.ad_net_fake = Discriminator()

    def forward(
        self,
        xs: torch.Tensor,
        norm_flag: bool = True,
    ):
        cls_outs, feats = self.net(xs, norm_flag=norm_flag)
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
        logits, _ = self(blobs, norm_flag=True)
        # features_real = features[labels == 1]
        # features_fake = features[labels == 0]
        # discriminator_outs_real = ad_net_real(features_real)
        # discriminator_outs_fake = ad_net_fake(features_fake)
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
