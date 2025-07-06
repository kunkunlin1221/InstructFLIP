from typing import Tuple

import torch
import torch.nn as nn

from ..base import BaseModelInterface
from .compenents import Classifier, FeatureEmbedder, FeatureGenerator


class ViT(BaseModelInterface):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        input_size: Tuple[int, int] = (224, 224),
        modality: str = "rgb",
    ):
        super().__init__()
        self._build_model(in_channels=in_channels, num_classes=num_classes, input_size=input_size)
        self.ce_loss = nn.CrossEntropyLoss()
        self.modality = modality

    def _build_model(self, in_channels, num_classes, input_size):
        self.backbone = FeatureGenerator(in_channels)
        x = torch.randn(1, in_channels, *input_size)
        feat_dim = self.backbone(x).shape[-1]
        self.embedder = FeatureEmbedder(feat_dim)
        self.head = Classifier(num_classes)

    def forward(self, xs, norm_flag: bool = True):
        feats = self.backbone(xs)
        feats = self.embedder(feats, norm_flag=norm_flag)
        cls_outs = self.head(feats, norm_flag=norm_flag)
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
